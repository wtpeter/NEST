"""Matrix-free ROKS and projected NTTDA response solves."""

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from nest.grad.nttda.roks import (
    _preconditioner, _response_reference, make_hessian_transpose_action,
)


def orbital_actions(tdobj, pairs):
    """Differentiate r=(F_beta_OC, (F_alpha+F_beta)_VC, F_alpha_VO).

    For an antisymmetric K, D'_s = K n_s - n_s K and
    F'_s = K.T F_s + F_s K + C.T v_s[D'] C.
    This is the forward Jacobian action, not an assembled orbital Hessian.
    """
    mf = tdobj._scf
    c = mf.mo_coeff
    occ = np.asarray((mf.mo_occ > 0, mf.mo_occ == 2), dtype=float)
    fock = mf.get_fock()
    f = np.asarray((c.T @ fock.focka @ c, c.T @ fock.fockb @ c))
    response = _response_reference(mf).gen_response(hermi=1)

    def forward(vector):
        k = np.zeros_like(f[0])
        for value, (p, q, _) in zip(vector, pairs):
            k[p, q], k[q, p] = value, -value
        dm = k[None] * (occ[:, None, :] - occ[:, :, None])
        potential = response(c @ dm @ c.T)
        derivative = k.T @ f + f @ k + c.T @ potential @ c
        fa, fb = derivative
        blocks = {'co': fb, 'cv': fa + fb, 'ov': fa}
        return np.array([blocks[name][p, q] for p, q, name in pairs])

    transpose, _ = make_hessian_transpose_action(tdobj, pairs)
    return forward, transpose, _preconditioner(tdobj, pairs)


def solve(action, rhs, diagonal, tolerance, max_cycle):
    """Restarted, diagonally preconditioned GMRES with an explicit residual check.

    At most 40 Krylov vectors are retained. No dense operator or inverse is
    constructed. The diagonal regularization affects only the preconditioner.
    """
    rhs = np.asarray(rhs)
    size = rhs.size
    if size == 0 or np.linalg.norm(rhs) <= tolerance:
        return np.zeros_like(rhs), float(np.linalg.norm(rhs)), 0
    diagonal = np.asarray(diagonal)
    safe = np.where(abs(diagonal) < 1e-4, np.where(diagonal < 0, -1e-4, 1e-4), diagonal)
    operator = LinearOperator((size, size), matvec=action, dtype=float)
    preconditioner = LinearOperator((size, size), matvec=lambda v: v / safe, dtype=float)
    iterations = 0

    def callback(_):
        nonlocal iterations
        iterations += 1

    solution, info = gmres(
        operator, rhs, M=preconditioner, atol=tolerance, rtol=0,
        restart=min(40, size), maxiter=max_cycle, callback=callback,
        callback_type='pr_norm',
    )
    residual = float(np.linalg.norm(action(solution) - rhs))
    if info != 0 or not np.isfinite(residual) or residual > tolerance:
        raise RuntimeError('NTTDA response GMRES failed: info=%d, residual=%.3g, tolerance=%.3g'
                           % (info, residual, tolerance))
    return solution, residual, iterations
