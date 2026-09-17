# SF-TDA / NT-TDA 的物理态溶剂能量

这些例子使用 `nest.solvent.StateSpecificPCM`。输入是普通气相 UKS/ROKS
参考和 NEST 的 TDA 对象；驱动用**指定物理根的密度**建立溶剂环境。
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
- `03_solvatochromism.py`：丙酮与对硝基苯胺的气相到水模型光谱位移。
  同时输出根间重叠，帮助跟踪同一个电子态。
- `04_ct.py`：文章 SI 中 5 Å 的乙烯/四氟乙烯二聚体，打印各根的片段电荷。
  先通过电荷变化辨认 CT 态，再比较其溶剂位移。

脚本开头设置了保守的线程默认值，外部可覆盖，最高 20。所有能量内部使用
Hartree，打印的能隙使用 eV。例子采用较小网格便于试算；定量工作应检查 DFT
积分网格、PCM 表面网格、基组和态求解的收敛性。

## 两个计算步骤

```python
field = pcm.PCM(mol)                  # 来自 pyscf.solvent
field.method = 'C-PCM'
field.eps = 78.3553                   # 静态介电常数
calculation = StateSpecificPCM(td, field, eps_optical=1.333**2)

equilibrium_energy = calculation.kernel(state=1)
vertical_gaps = calculation.vertical(method='ptss')
```

`state` 使用 **1-based** 编号，表示气相 TDA 谱中的出发根。外循环按振幅重叠
跟踪这个根，`calculation.state` 保存最终的能量排序编号。

`kernel` 在当前几何上使该物理态与静态溶剂平衡，返回其模型总能量。
`vertical` 保留这一慢溶剂环境，返回各目标态相对出发态的能隙；出发态自身
对应的能隙接近零。`vertical(other_td)` 可计算同一个 SCF 对象上其他自旋
分支的目标态，例如在 S0 环境中计算 T1。为研究 T1 环境中的跃迁，应另建
驱动，以 T1 为出发态进行 `kernel`。

结果字段：

| 字段 | 含义 |
|---|---|
| `e_tot` | 出发物理态的平衡模型总能量 |
| `dm`, `v` | 出发态 AO 电荷密度与反应势 |
| `gas_td` | 出发分支的气相 TDA 结果 |
| `td` | 最近一次固定反应场求解的振幅与本征值 |
| `e_frozen` | 公共反应场近似下各目标态的总能量 |
| `e_ptss`, `ptss_correction` | 最近一次 ptSS 总能量及修正 |
| `e_ptlr`, `ptlr_correction` | 最近一次 ptLR 总能量及修正 |
| `cycles`, `history` | 外循环次数及每轮能量、密度变化、跟踪重叠 |

每次 `vertical` 清空上一次修正结果，只保留本次选定方法的结果字段。
`td.e` 保存固定场本征值；用 `vertical()` 的返回值读取最终物理能隙。
修正后的能量按固定场根编号返回，便于保留态的身份；它们可能出现新的能量排序。
`td` 可用于态分析。其梯度、NAC 和显式 A/B 构造目前会明确报错。

## 本实现的能量定义

每个几何上的参考轨道和气相 TDA 算符 $A_0$ 固定。对归一化物理态振幅 $x$，
采用模型泛函

$$G[x] = E_{\rm ref} + x^\dagger A_0 x + E_{\rm PCM}[D[x]].$$

$D[x]$ 是包含参考占据的自旋求和波函数密度。反应势通过自旋标量一电子
矩阵元作用于完整 TDA 振幅空间，外循环更新振幅和 PCM 极化。这个定义使能量
计数与密度更新一致，也允许复用昂贵的气相响应核。参考轨道在此外循环中保持固定；
轨道弛豫属于后续模型扩展。

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

**ptLR** 是对上述物理态模型作线性响应得到的、孤立跃迁极点的一阶修正：

$$\Delta\omega_k^{\rm ptLR}=
\operatorname{Tr}(D_{Ak} B_\infty[D_{kA}]).$$

这里使用两个**物理根之间**的跃迁密度。它可以描述同一 NT 自旋分支中的
电荷跃迁；不同总自旋的纯 NT 态之间，标量电荷跃迁矩阵元为零。SF 的不同
磁量子数分支之间同样为零。近简并跃迁需要求解耦合响应问题，目前 ptLR
只作为孤立根的一阶近似。测试通过非平衡能量的二阶差分检查其符号和系数。

`ptss` 与 `ptlr` 是两种分别报告的近似，当前驱动分别求值。自洽 SS 的比较
可通过另建驱动并选择另一个物理根实现。平衡态总能量之差和非平衡垂直能隙
对应不同的溶剂条件，例子会给出明确标签。

## 如何判断验证结果

先检查电子数、态密度和一电子算符矩阵元，再检查能量恒等式与气相极限。
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

| 例子与所比较的态 | 气相 / eV | 水模型 ptSS / eV | 位移 / eV |
|---|---:|---:|---:|
| 丙酮 SF-TDA，最低单重激发态候选 | 4.17665 | 4.55553 | +0.37888 |
| 对硝基苯胺 SF-TDA，强吸收单重态候选 | 4.45716 | 3.97053 | -0.48663 |
| SI 二聚体 NT-TDA，C2F4 电荷变化约 +1e | 14.98117 | 13.55180 | -1.42937 |
| SI 二聚体 NT-TDA，C2F4 电荷变化约 -1e | 15.54877 | 14.36547 | -1.18330 |

丙酮所选根的气相自旋为 `<S²>=0.0163`，气相到溶液的振幅重叠为 0.9968。
对硝基苯胺所选根的气相自旋为 `<S²>=0.1742`，振子强度为 0.8397；它从气相
第 5 根变为溶液第 3 根，振幅重叠为 0.9910。按物理态重叠跟踪保留了这个对应关系。
CT 二聚体使用 HF/STO-3G 和 16 根搜索；两个 CT 根的振幅重叠均接近 1。
这些数字提供可复现的定性检查，定量光谱精度还需更充分的基组、网格和模型比较。

甲醛 NT-TDA 例子中，S0 自洽 9 轮；在同一环境中得到的 ptSS S1、T1
能隙分别为 3.88835、3.65663 eV。SF-TDA 例子自洽 7 轮，5 个根的密度积分
均为 16 个电子；将基组增大为 def2-TZVP 后也在 7 轮收敛，单重激发态候选
的能隙从 3.99378 变为 3.95779 eV。丙酮 SF-TDA 的完整流程自洽 7 轮，耗时约 93 秒；另一项
丙酮 NT-TDA 成本探针中，气相 5 根求解约需 224 秒。机器上存在其他并行任务，
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

上述回归得到 **45 个测试及 13 个子测试通过**，Ruff 检查通过。密度测试包含
显式行列式展开与费米产生/湮灭算符的独立比较，覆盖 NT 的三个自旋分支及
SF 的两个翻转方向；能量测试覆盖气相极限、共同溶剂环境、ptSS 二次展开和
ptLR 的能量曲率。
共同环境测试还检查了以三重态建立独立平衡环境，再计算通往较低单重态的负能隙。

## 几何与文献来源

- `geometries/acetone.xyz` 和 `geometries/p-nitroaniline.xyz`：PubChem
  3D conformer，2026-09-17 获取；完整请求地址保留在 XYZ 注释行。
  这些是演示用的固定几何。
- `geometries/c2h4_c2f4_5A.xyz`：Alam 等的补充材料原始坐标。
- [Alam et al., JCP 156, 194110 (2022)](https://doi.org/10.1063/5.0091636)：
  物理态自洽和共享出发态环境的 ptSS 流程。论文使用 Marcus 分割及相应极化功项；
  本实现以上述共同环境展开定义非平衡能量，文献算例用于方法比较。
- [p-Nitroaniline solvatochromism, J. Phys. Chem. A](https://doi.org/10.1021/jp110026c)。
- [Acetone solvatochromism](https://research.rug.nl/en/publications/solvatochromism-of-the-%CF%80-n-transition-of-acetone-by-combined-quan/)。

解析梯度、几何优化、SMD、完整 SF-TDDFT 和 DZ0SCF 留待后续阶段。
