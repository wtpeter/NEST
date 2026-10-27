# SF-TDA / NT-TDA 的物理态溶剂能量

这些例子使用 `nest.solvent.StateSpecificPCM`。输入是已收敛的普通 vacuum UKS/ROKS
参考和 NEST 的 TDA 模板；驱动用**指定物理根的密度**建立溶剂环境。
SCF 使用真空边界条件，TDA 模板只需设置自旋分支、根数等参数。首次 TDA 求解
由溶剂驱动完成，调用者可以省去 `vacuum_td.kernel()`。
第一阶段提供固定几何的能量，适用于实轨道、C-PCM、SF-TDA 和 NT-TDA。
当前使用 PySCF 的 CPU 后端。

## 从哪个例子开始

在 `nest` Conda 环境中，从仓库根目录运行，例如：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python examples/solvent/01_nttda.py
```

- `01_nttda.py`：甲醛的 NT-TDA；先与最低单重态平衡，再在同一环境中计算
  单重态和三重态的垂直能隙。展示 `frozen`、`ptss`、`ptlr` 三种结果。
- `02_sftda.py`：UKS 参考的 SF-TDA；展示物理根的电子数、自旋和能隙。
- `03_solvatochromism.py`：丙酮与对硝基苯胺的真空到水模型光谱位移。
  同时输出根间重叠，帮助跟踪同一个电子态。
- `04_ct.py`：文章 SI 中 5 Å 的乙烯/四氟乙烯二聚体，打印各根的片段电荷。
  先通过电荷变化辨认 CT 态，再比较其溶剂位移。
- `05_ss_validation.py`：甲醛与 CT 二聚体，在相同慢溶剂环境中比较 ptSS
  和目标态自洽 SS，并比较直接迭代与 DIIS；CT 目标显式使用 `follow_state=True`。
- `06_lr_validation.py`：实验性 ptLR 的模型内部检查；完整小模型空间的耦合 LR，
  及人为近简并两能级的混合。

第二批的模型核对、独立验证及误差表见 [VALIDATION.md](VALIDATION.md)。

脚本开头设置了保守的线程默认值，外部可覆盖，最高 20。所有能量内部使用
Hartree，打印的能隙使用 eV。例子采用较小网格便于试算；定量工作应检查 DFT
积分网格、PCM 表面网格、基组和态求解的收敛性。

## 输入与计算顺序

```python
# mf 是已经收敛的真空 ROKS；这里以三重态参考的单重态分支为例。
vacuum_td = mf.NTTDA().set(deltaS=-1, nstates=5)
field = pcm.PCM(mol)                  # 来自 pyscf.solvent
field.method = 'C-PCM'
field.eps = 78.3553                   # 静态介电常数
calculation = StateSpecificPCM(vacuum_td, field, eps_optical=1.333**2)

equilibrium_energy = calculation.kernel(state=1)       # 平衡 SS：快、慢极化与最低根平衡
vertical_gaps = calculation.vertical().copy()         # 默认 ptSS，返回整批垂直能隙
ss_gap = calculation.state_specific(state=2)          # 非平衡 SS：固定出发态慢极化
```

`state` 使用 **1-based** 编号。`kernel` 和 `state_specific` 默认在**每一轮按
本征值升序选择第 `state` 个物理根**。`state=1` 表示当前自旋分支的最低根。
上一轮振幅用于求解初猜，overlap 记录相邻迭代之间的变化。

需要保持某个 CT 态的身份时，可用 `kernel(state=k, follow_state=True)` 或
`state_specific(state=k, follow_state=True)`。此时 `k` 分别指初始真空谱或
初始共同场谱的根，后续按最大振幅重叠跟踪；`state` / `ss_state` 记录最终排序。
这是相对于早期版本的默认行为变化，旧的跟踪计算应显式加入该参数。

`kernel` 在当前几何上使该物理态与静态溶剂平衡，返回其模型总能量。
`vertical` 保留这一慢溶剂环境，返回各目标态相对出发态的能隙；出发态自身
对应的能隙接近零。`vertical(other_td)` 可计算同一个 SCF 对象上其他自旋
分支的目标态，例如在 S0 环境中计算 T1。为研究 T1 环境中的跃迁，应另建
驱动，以 T1 为出发态进行 `kernel`。

`state_specific` 使用已经平衡的出发态环境，因此先运行 `kernel`。
它只让目标振幅和快极化自洽。计算目标态自身的平衡能量时，另建驱动并调用
`kernel(state=...)`，让快、慢极化都与目标态平衡。

输入和结果对象的对应关系：

| 字段 | 含义 |
|---|---|
| 输入 `vacuum_td` | 真空 TDA 模板，提供方法设置和已收敛的 SCF 参考 |
| `base` | 驱动保存的输入模板副本，与输入共享同一个 SCF 参考 |
| `e_tot` | 出发物理态的平衡模型总能量 |
| `dm`, `v` | 出发态 AO 电荷密度与反应势 |
| `gas_td` | 初次求得的真空 TDA 结果；沿用已有公开字段名 |
| `td` | 最近一次固定反应场求解的振幅与本征值 |
| `e_frozen` | 公共反应场近似下各目标态的总能量 |
| `e_ptss`, `ptss_correction` | 最近一次 ptSS 总能量及修正 |
| `e_ptlr`, `ptlr_correction` | 最近一次实验性 ptLR 总能量及修正 |
| `cycles`, `history` | 外循环次数及每轮能量、密度变化、相邻迭代振幅重叠 |
| `e_ss`, `ss_dm`, `ss_td`, `ss_state` | 最近一次非平衡目标态 SS 的总能量、密度、谱及最终根编号 |
| `ss_converged`, `ss_cycles`, `ss_history` | 非平衡目标态 SS 的收敛状态、轮数和历史 |

每次 `vertical` 清空上一次修正结果，只保留本次选定方法的结果字段。
`calculation.td.e` 保存固定场本征值；用 `vertical()` / `state_specific()` 的返回值
读取最终物理能隙。`state_specific` 会刷新共同场结果，并清空此前的 ptSS/ptLR
字段；比较各方法时分别保存返回值，如上面的 `.copy()`。
修正后的能量按固定场根编号返回，便于保留态的身份；它们可能出现新的能量排序。
`calculation.td` 可用于态分析。`_FrozenFieldTDA` 为它加上固定场结果的类型标记，
并限制尚未实现的梯度、NAC 和显式 A/B 接口；外层循环仍可在下一轮更新反应场。

## 本实现的能量定义

每个几何上的参考轨道和真空 TDA 算符 $A_0$ 固定。对归一化物理态振幅 $x$，
采用模型泛函

$$G[x] = E_{\rm ref} + x^\dagger A_0 x + E_{\rm PCM}[D[x]].$$

$D[x]$ 是包含参考占据的自旋求和波函数密度。反应势通过自旋标量一电子
矩阵元作用于完整 TDA 振幅空间，外循环更新振幅和 PCM 极化。这个定义使能量
计数与密度更新一致，也允许复用昂贵的真空响应核。参考轨道在此外循环中保持固定；
轨道弛豫属于后续模型扩展。

内层统一使用 `nest._lr_eig.eigh`。NT-down 的 OO 单位矩阵方向在初猜、完整
算符和预条件器中投影掉，与普通 NT-TDA 共用 `_project_sfd`。求解初猜包含上一轮振幅
和低对角元方向，以便在态重排时发现新的低能根。

对于与物理态 A 平衡的环境，固定场中的目标态 k 使用以下共同环境展开：

$$G_k^{\rm frozen}=E_{\rm ref}+x_k^\dagger A_0x_k+
E_{\rm PCM}[D_A]+\operatorname{Tr}(V_A(D_k-D_A)).$$

驱动在本征值中已经计入反应势，形成总能量时会相应扣除已计入的线性项。

**ptSS** 对该固定场波函数计算快极化的二次修正：

$$\Delta G_k^{\rm ptSS}=\tfrac12\operatorname{Tr}
\left[(D_k-D_A)B_\infty[D_k-D_A]\right].$$

$B_\infty$ 是与同一空腔、光学介电常数 $\epsilon_\infty=n^2$ 对应的 PCM
密度响应。非平衡模型以静态场 $V_A$ 为展开点，慢场等于静态反应场减去同一
密度的光学反应场（Pekar 分割）。上面的能量表达式完整定义了本实现。
当静态与光学介电常数相等时，二次展开精确恢复同一波函数密度的平衡 PCM
能量；代码测试直接检查这个恒等式。

**ptLR（实验性）** 是对上述物理态模型作线性响应得到的、孤立跃迁极点的一阶修正：

$$\Delta\omega_k^{\rm ptLR}=
\operatorname{Tr}(D_{Ak} B_\infty[D_{kA}]).$$

这里使用两个**物理根之间**的跃迁密度。它可以描述同一 NT 自旋分支中的
电荷跃迁；不同总自旋的纯 NT 态之间，标量电荷跃迁矩阵元为零。SF 的不同
磁量子数分支之间同样为零。近简并跃迁需要求解耦合响应问题，目前 ptLR
只作为孤立根的一阶近似。测试通过非平衡能量的二阶差分检查其符号和系数。
这些检查验证模型内部的数学一致性；独立程序、实验和激发出发态的系统基准
仍待补充。实际计算建议以 ptSS 为起点，用自洽 SS 抽查振幅弛豫的影响。

`ptss` 与 `ptlr` 是两种分别报告的近似，当前驱动分别求值。自洽 SS 的比较
可使用 `state_specific(state=2)`：每轮取第二根，该方法固定出发态慢极化，
只迭代目标态及快极化。返回相对于出发态的非平衡 SS 能隙。
目标态与整个溶剂平衡的能量可通过另建驱动并选择该物理根执行 `kernel` 获得。
`diis_space=6` 可加速这两个外循环；默认值 0 保持直接迭代。
外循环求解密度固定点 $D_{out}=F(D_{in})$，DIIS 使用残差 $D_{out}-D_{in}$。
能量和密度同时达到阈值才报告收敛；可通过直接迭代核对所得解。

## 如何判断验证结果

先检查电子数、态密度和一电子算符矩阵元，再检查能量恒等式与真空极限。
小体系用自洽 SS 对比 ptSS；响应减弱时检查二者接近的程度。

物理趋势候选包括丙酮的 $n\to\pi^*$ 蓝移、对硝基苯胺的 CT 型
$\pi\to\pi^*$ 红移，以及片段间 CT 的距离依赖。先按重叠、振子强度和
电荷分布识别同一电子态，再比较位移。实验溶剂位移还包含显式氢键、溶剂
构型采样等贡献；这里验证的是固定几何、连续介质下的电子极化模型。

C-PCM 的半径、缩放和表面网格同样属于模型参数。SI 的 CT 几何为 **5 Å**；
Alam 等文章 Table I 的几何为 **4.5 Å**。定量对照应统一几何和参数，并把
电子结构方法造成的绝对能量差异与溶剂位移分别报告。

## 已运行的验证

验证环境为 PySCF 2.14.0、NumPy 2.4.6，所有计算使用 2 线程。
上述 CAM-B3LYP 例子使用密度拟合、def2-SVP、DFT 网格 level 2，PCM 的
`lebedev_order=11`；水参数为 `eps=78.3553`、`eps_optical=1.333**2`。

| 例子与所比较的态 | 真空 / eV | 水模型 ptSS / eV | 位移 / eV |
|---|---:|---:|---:|
| 丙酮 SF-TDA，最低单重激发态候选 | 4.17665 | 4.55553 | +0.37888 |
| 对硝基苯胺 SF-TDA，强吸收单重态候选 | 4.45716 | 3.97053 | -0.48663 |
| SI 二聚体 NT-TDA，C2F4 电荷变化约 +1e | 14.98117 | 13.55180 | -1.42937 |
| SI 二聚体 NT-TDA，C2F4 电荷变化约 -1e | 15.54877 | 14.36547 | -1.18330 |

丙酮所选根的真空自旋为 `<S²>=0.0163`，真空到溶液的振幅重叠为 0.9968。
对硝基苯胺所选根的真空自旋为 `<S²>=0.1742`，振子强度为 0.8397；它从真空
第 5 根变为溶液第 3 根，振幅重叠为 0.9910。按物理态重叠跟踪保留了这个对应关系。
CT 二聚体使用 HF/STO-3G 和 16 根搜索；两个 CT 根的振幅重叠均接近 1。
这些数字提供可复现的定性检查，定量光谱精度还需更充分的基组、网格和模型比较。

甲醛 NT-TDA 例子中，S0 自洽 9 轮；在同一环境中得到的 ptSS S1、T1
能隙分别为 3.88835、3.65663 eV。SF-TDA 例子自洽 7 轮，5 个根的密度积分
均为 16 个电子；将基组增大为 def2-TZVP 后也在 7 轮收敛，单重激发态候选
的能隙从 3.99378 变为 3.95779 eV。丙酮 SF-TDA 的完整流程自洽 7 轮，耗时约 93 秒；另一项
丙酮 NT-TDA 成本探针中，真空 5 根求解约需 224 秒。机器上存在其他并行任务，
这些耗时用于估计量级。16 原子的对硝基苯胺 SF-TDA 完整流程自洽 11 轮，
耗时约 770 秒；这个规模下逐个目标态重复 SS 的成本已经显著，批量 ptSS 更适合初筛。

另外，水分子/HF/STO-3G 的数值检查显示：

- 静态和光学介电常数同时取 1.2、1.1、1.05 时，激发态 ptSS 与自洽 SS
  总能量的差依次为 `2.47e-6`、`7.34e-7`、`2.01e-7` Hartree。
- `eps=10`、`eps_optical=2` 时，PCM 表面阶数从 23 增加至 29，前两个
  非零能隙分别变化约 0.00039 和 0.00030 eV。这也说明演示用的阶数 11
  应在定量工作中继续加密。

回归命令：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    python -m pytest src/nest/solvent/tests src/nest/sftda/tests src/nest/nttda/tests -q
python -m ruff check .
```

第一批回归得到 **45 个测试及 13 个子测试通过**，Ruff 检查通过。第二批增加
非平衡 SS 与耦合 LR 验证后，共 **51 个测试及 15 个子测试通过**。密度测试包含
显式行列式展开与费米产生/湮灭算符的独立比较，覆盖 NT 的三个自旋分支及
SF 的两个翻转方向；能量测试覆盖真空极限、共同溶剂环境、ptSS 二次展开和
ptLR 的能量曲率。
共同环境测试还检查了以三重态建立独立平衡环境，再计算通往较低单重态的负能隙。

## 几何与文献来源

- `geometries/acetone.xyz` 和 `geometries/p-nitroaniline.xyz`：PubChem
  3D conformer，2026-09-17 获取；完整请求地址保留在 XYZ 注释行。
  这些是演示用的固定几何。
- `geometries/c2h4_c2f4_5A.xyz`：Alam 等的补充材料原始坐标。
- [Alam et al., JCP 156, 194110 (2022)](https://doi.org/10.1063/5.0091636)：
  物理态自洽和共享出发态环境的 ptSS 流程。论文使用 Marcus 分割及相应极化功项；
  本实现使用 Pekar 共同环境展开，两者的 C-PCM 总极化自由能等价，推导与数值
  核对见 [VALIDATION.md](VALIDATION.md)。文献算例用于方法比较。
- [p-Nitroaniline solvatochromism, J. Phys. Chem. A](https://doi.org/10.1021/jp110026c)。
- [Acetone solvatochromism](https://research.rug.nl/en/publications/solvatochromism-of-the-%CF%80-n-transition-of-acetone-by-combined-quan/)。

溶剂解析梯度、几何优化、SMD、完整 SF-TDDFT 和平均占据参考 AOCSCF 留待后续阶段。
