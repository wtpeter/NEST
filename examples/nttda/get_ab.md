# Building an NTTDA Result from `get_ab`

`NTTDA.get_ab()` returns the explicit TDA A matrix as a NumPy array. The
matrix uses the same flattened amplitude ordering as `gen_vind_*`, so a full
diagonalization can be used to construct an object with the same result fields
as an `NTTDA.kernel()` calculation.

```python
import cupy as cp
import numpy as np

from nest.gpu import nttda


def diagonalize_nttda(mf, deltaS, nstates=5, nobeta=False):
    if deltaS not in (-1, 0, 1):
        raise ValueError("deltaS must be -1, 0, or 1")

    fake_td = nttda.NTTDA(mf).set(
        deltaS=deltaS,
        nstates=nstates,
        nobeta=nobeta,
    )
    a = fake_td.get_ab()
    energies, amplitudes = np.linalg.eigh(a)

    # The deltaS=-1 space contains one spin-lowering zero root.
    if deltaS == -1:
        keep = np.abs(energies) > 1e-8
        energies = energies[keep]
        amplitudes = amplitudes[:, keep]

    energies = energies[:nstates]
    amplitudes = cp.asarray(
        amplitudes[:, :nstates].T,
        dtype=mf.mo_coeff.dtype,
    )

    occupation = cp.asarray(mf.mo_occ)
    nclosed = int(cp.count_nonzero(occupation == 2).get())
    nopen = int(cp.count_nonzero(occupation == 1).get())
    nvirtual = int(cp.count_nonzero(occupation == 0).get())

    if deltaS == 1:
        xy = [
            (vector.reshape(nclosed, nvirtual), 0)
            for vector in amplitudes
        ]
    elif deltaS == -1:
        xy = [
            (
                vector.reshape(
                    nclosed + nopen,
                    nopen + nvirtual,
                ),
                0,
            )
            for vector in amplitudes
        ]
    else:
        xy = [(vector, 0) for vector in amplitudes]

    fake_td.e = energies
    fake_td.xy = xy
    fake_td.converged = cp.ones(len(energies), dtype=cp.bool_)
    fake_td.nstates = len(energies)
    return fake_td
```

For a density-fitted calculation, construct the mean-field object with
`density_fit()` before calling the function:

```python
mf = mol.ROKS(xc="HYB_MGGA_X_M06_SX,MGGA_C_M06_SX").to_gpu()
mf = mf.density_fit().run()
td = diagonalize_nttda(mf, deltaS=-1, nstates=5)
```

The returned object provides the result fields normally consumed after
`kernel()`:

- `td.e`: NumPy array of excitation energies.
- `td.xy`: list of `(X, 0)` pairs with CuPy amplitudes in the native shape.
- `td.converged`: CuPy boolean array.
- `td.nstates`: number of retained roots.

This constructs the numerical result directly. It does not reproduce Davidson
iteration history, logging, or checkpoint side effects from `kernel()`.
