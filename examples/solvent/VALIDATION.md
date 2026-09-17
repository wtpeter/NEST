# 物理态 PCM 能量模型核对与第二批验证

本页记录固定轨道模型的定义、独立验证和近似误差。可运行程序为
`05_ss_validation.py` 和 `06_lr_validation.py`；回归测试为
`src/nest/solvent/tests/test_nonequilibrium.py`。

## 模型与文献的对应关系

NEST 以固定参考轨道上的气相 TDA 矩阵 $A_0$ 作为有效电子算符。物理态振幅
$x$ 满足 $x^\dagger x=1$，其电荷密度 $D[x]$ 包含参考占据。平衡能量为

$$G[x]=E_{\rm ref}+x^\dagger A_0x+E_{\rm PCM}[D[x]].$$

这个定义保留了 RAS-SF 文献中“用物理态密度建立反应场，再更新物理态”的
思路。NT/SF-TDA 的 $A_0$、固定参考轨道和波函数密度共同定义当前近似。
其中，泛函响应核在自洽外循环中固定；轨道弛豫和能量导数密度属于后续模型扩展。

标量势的矩阵元素由完整 $E_{pq}$ 密度给出。代码使用相对于参考占据正规排序的
势矩阵 $P(V)$，因此其本征值满足

$$\lambda=x^\dagger A_0x+\operatorname{Tr}\{V(D[x]-D_{\rm ref})\}.$$

总能量中用此式恢复 $x^\dagger A_0x$，再加入 PCM 极化能。这个计数由
气相极限、完整密度的显式行列式验证，以及独立驻点方程共同检查。

## Marcus 与 Pekar 分割

[You et al., JCP 143, 204104 (2015)](https://doi.org/10.1063/1.4936357)
讨论了两种分割及离散化。对于本实现使用的 C-PCM，表面响应矩阵对称，完整
非平衡自由能可在两种分割中保持一致。
[Alam et al., JCP 156, 194110 (2022)](https://doi.org/10.1063/5.0091636)
使用 Marcus 分割，并在式 (21)–(23) 中包含快慢表面电荷的交叉极化功。

以下给出当前 C-PCM 离散模型的核对。用 $S$ 表示表面电荷库仑矩阵，$v_0,v_k$
表示初态、末态的总溶质表面势（含核电荷），并定义

$$f_0=\frac{\epsilon_0-1}{\epsilon_0},\quad
f_\infty=\frac{\epsilon_\infty-1}{\epsilon_\infty},\quad
q_0=-f_0S^{-1}v_0.$$

Marcus 分割的慢电荷与目标快电荷可写为

$$q_s=\frac{\epsilon_0-\epsilon_\infty}{\epsilon_0-1}q_0,\qquad
q_{f,k}=-f_\infty S^{-1}(v_k+Sq_s).$$

将交叉功 $\tfrac12(q_{f,k}-q_{f,0})^TSq_s$ 计入后，极化自由能为

$$G_{\rm pol}^{M}=v_k^T(q_s+q_{f,k})
-\tfrac12v_0^Tq_s-\tfrac12v_k^Tq_{f,k}
+\tfrac12(q_{f,k}-q_{f,0})^TSq_s.$$

代入上面的表面电荷，得到与代码共同环境展开一致的表达式：

$$G_{\rm pol}^{P}=\tfrac12v_0^Tq_0+(v_k-v_0)^Tq_0
-\tfrac12f_\infty(v_k-v_0)^TS^{-1}(v_k-v_0).$$

测试用 PySCF 的实际 C-PCM 空腔和三个物理根的密度分别计算这两式，一致到
$10^{-12}$ Hartree。这里的等价性依赖对称的 C-PCM 响应；其他 PCM 变体需要
单独核对其离散算符。

## 相同慢环境下的 ptSS 与自洽 SS

设出发态 A 已经平衡，固定其 $D_A$、$V_A$ 和慢极化。目标态的非平衡泛函为

$$G_A[x]=E_{\rm ref}+x^\dagger A_0x+E_{\rm PCM}[D_A]
+\operatorname{Tr}(V_A\Delta D)
+\tfrac12\operatorname{Tr}(\Delta D B_\infty[\Delta D]),\qquad
\Delta D=D[x]-D_A.$$

ptSS 在共同反应场的波函数上计算此泛函。新增的 `state_specific` 则迭代目标态
振幅与势 $V_A+B_\infty[\Delta D]$，直到能量和密度同时收敛。比较两者时，
几何、参考轨道、空腔和慢极化完全相同；误差反映这一定义下目标振幅弛豫的影响。
激发态对应驻点，误差的符号由具体态及其混合决定。

```python
calculation.kernel(state=1)                   # 建立出发态 A 的平衡环境
ptss = calculation.vertical().copy()
ss_gap = calculation.state_specific(state=2)   # 共同场谱的第 2 根，1-based
print(ptss[1], ss_gap)                        # 都相对于 A，单位 Hartree
print(calculation.ss_cycles, calculation.ss_converged)
```

`state_specific` 保留出发态 `dm`、`v`、`e_tot`、`state`。目标结果存于 `e_ss`、
`ss_dm`、`ss_td`、`ss_state`、`ss_history`；`ss_td` 是最后一次目标反应场下的
谱。初始化会刷新共同场谱并清空此前的 ptSS/ptLR 修正字段；比较时先保存返回的
能隙，如上例中的 `ptss.copy()`。共同场的跃迁性质应使用 `vertical` 生成的 `td`。对不同目标分别自洽得到的
波函数需要另外处理它们之间的非正交性。

独立验证包括：

- $\epsilon_\infty=1$ 时，SS 回到共同固定场结果。
- $\epsilon_\infty=\epsilon_0$ 时，SS 与独立平衡的目标态一致。
- 同一个出发态自身的垂直能隙为零；目标计算保持出发态环境不变。
- 在水/HF/STO-3G 的完整 17 根物理空间里，直接用态间密度矩阵元组装非线性
  驻点方程，通过 SciPy 求根获得独立解；与迭代求解器的总能量比较至
  $10^{-9}$ Hartree，密度比较至 $2\times10^{-8}$。
- 覆盖 SF 的 `col`、`mcol`，以及 NT 单重态环境中的三重态目标。

## ptLR 与耦合 LR

对同一个非平衡泛函在出发态附近作小振幅展开。用完整共同场物理根作基，设
$\Delta_k$ 是固定场能隙，$D_{Ak}$ 是物理根之间的跃迁密度，并定义

$$K_{kl}=\operatorname{Tr}(D_{Ak}B_\infty[D_{lA}]).$$

在实轨道、实驻态振幅下，实部和虚部变化的能量 Hessian 分别为
$H_q=2\operatorname{diag}(\Delta)+4K$、$H_p=2\operatorname{diag}(\Delta)$。
因此响应矩阵为

$$A_{\rm LR}=\operatorname{diag}(\Delta)+K,\qquad B_{\rm LR}=K,\qquad
\mathcal L=\begin{pmatrix}A_{\rm LR}&B_{\rm LR}\\-B_{\rm LR}&-A_{\rm LR}\end{pmatrix}.$$

孤立根一阶展开给出代码中的 $\omega_k^{\rm ptLR}=\Delta_k+K_{kk}$。
测试独立地对复振幅的实部、虚部做二阶差分，恢复整个 A/B 矩阵并比较耦合谱；
快响应缩放一半时，最低孤立根的一阶截断误差近似缩小到四分之一。

`06_lr_validation.py` 使用完整的 17 根小模型空间。它还把实际核中的两个耦合
方向放进人为近简并的两能级模型，展示非对角耦合引起的模混合。大体系只取少数
根时，类似矩阵具有额外的态空间截断误差。当前公共能量接口继续提供 ptLR；
完整耦合 LR 在本批作为验证程序使用。

## 收敛加速

`diis_space=6` 启用 PySCF DIIS，使用输入、输出密度之差作为残差，并同时用于
出发态平衡和目标态非平衡 SS。默认 `diis_space=0` 保持直接迭代，便于比较。
`05_ss_validation.py` 在同一模型、同一阈值下比较两种迭代，检查最终能量一致。

## 已运行的结果（2026-09-18）

运行环境为 `nest`、PySCF 2.14.0、NumPy 2.4.6，OMP、MKL、OpenBLAS 均为
2 线程。甲醛使用 CAM-B3LYP/def2-SVP、密度拟合及 DFT 网格 level 2；CT
二聚体使用 HF/STO-3G 和 SI 的 5 Å 几何。二者均用水参数
`eps=78.3553`、`eps_optical=1.333**2`、PCM 表面阶数 11。

| 共同场的目标根 | ptSS / eV | 非平衡 SS / eV | ptSS−SS / eV | 波函数重叠 |
|---|---:|---:|---:|---:|
| 甲醛，第 2 根 | 3.88834737 | 3.88712008 | +0.00122729 | 0.999929 |
| CT 二聚体，第 10 根 | 13.55179733 | 13.55116464 | +0.00063269 | 0.999295 |
| CT 二聚体，第 15 根 | 14.36547420 | 14.36460195 | +0.00087225 | 0.999964 |

CT 目标在自洽后分别排序为第 5、6 根，跟踪保持了对应关系。这批结果支持在
这些固定轨道模型中使用 ptSS 近似。CT 的小基组限制了可用极化自由度；对更大
基组、强态混合或其他构型，应继续用 `state_specific` 抽查误差。

| 计算部分 | 直接迭代轮数 | DIIS 轮数 |
|---|---:|---:|
| 甲醛出发态 | 9 | 5 |
| 甲醛目标态 | 5 | 4 |
| CT 出发态 | 3 | 3 |
| CT 第 10 根 | 5 | 4 |
| CT 第 15 根 | 3 | 3 |
| 对硝基苯胺出发态（SF-TDA，16 原子） | 11 | 5 |

两种迭代的总能量及 SS 能隙差均小于 $10^{-8}$ Hartree。甲醛完整流程耗时
147.0→112.1 秒，CT 完整流程 20.1→19.1 秒；这些是同时存在其他任务时的
实测值，轮数和能量一致性提供了更直接的比较。

另外按 `03_solvatochromism.py` 的对硝基苯胺设置启用 `diis_space=6`，完整流程
耗时 499.5 秒，第一批直接迭代为 769.8 秒。所选气相第 5 根仍对应溶液第 3 根，
振幅重叠为 0.990971；ptSS 能隙仍为 3.97053 eV，气相到水模型的位移仍为
−0.48663 eV。这项成本比较使用了相同计算设置，机器上的其他负载可能不同。

小模型完整耦合 LR 与 ptLR 按谱排序比较的最大能量差为：

| 快响应核缩放 | 最大谱差 / eV |
|---:|---:|
| 1 | 0.001854724 |
| 0.1 | 0.000018327 |
| 0.01 | 0.000000183 |

响应减弱时呈预期的二阶截断误差趋势。人为近简并两能级中，ptLR 给出
40.310688、40.417010 eV，耦合 LR 给出 40.295239、40.432197 eV，两个原始
方向的前向分量归一权重约为 89%/11%。这些人为能级用于展示耦合效应。

相关回归：**51 个测试、15 个子测试通过**；Ruff 检查通过。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python -m pytest src/nest/solvent/tests src/nest/sftda/tests src/nest/nttda/tests -q
```

## 梯度范围

本批验证只需要能量和密度。梯度后续需要把参考轨道响应、有效 TDA 算符导数、
PCM 空腔导数和所选快慢溶剂约束纳入同一个能量定义；届时可复用现有 NT/SF
梯度实现中的对应部分。
