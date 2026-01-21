# One-click correctness validation (1D L=9): `original` vs `hybrid-svd`

本目录包含一次**小尺寸正确性验证**的输出图与对应的原始数据（`*_data.npz`），用于比较：

- **original**：传统实现（不使用 checkpoint / hybrid）
- **hybrid-svd**：完全体 hybrid（checkpoint + 轨迹压缩使用 Hybrid-SVD/rSVD）

验证 run：`/root/zyk/PPI/PPI-Quantum_Dynamics/test/oneclick_validate/20260121-010415`

---

## 为什么要做这个测试？

Hybrid-SVD 的目标是在保持物理结果一致的前提下，降低存储/内存（通过 checkpoint + 压缩）并支持更大的系统或更长的 flow。

因此我们需要一个**最小可复现**、**一键**的正确性对比：

- **固定同一个 disorder realization**：避免因为随机势不同导致“看起来差很多”的假差异。
- **比较最终输出的关键物理对象**：哈密顿量对角部分、相互作用张量、LIOM（局域守恒量）等。
- **对 LIOM 使用 gauge-invariant 指标**：LIOM 的表示存在符号/排序/近简并子空间旋转等自由度，直接逐元素比较会夸大差异。

---

## 测试配置（来自 `summary.json`）

- **系统**：1D, `L=9`（`n=9`）
- **disorder**：`random`, `d=1.0`, `p=0`
- **cutoff**：`1e-4`
- **ODE tol**：`1e-8`（rtol=atol）
- **hybrid flow horizon**：`lmax_hybrid=6000`
- **Hybrid-SVD**：
  - `rank_h2=64`
  - `rank_h4=81`（对 `n=9` 时的 `81×81` reshape 接近“满秩”）
  - `store_dtype=float32`, `niter=2`, `oversample=16`

---

## 这次测试看哪些量？为什么？

下面每一项都包含两部分信息：

- **如何计算/比较**：这张图在脚本里具体对哪些数组做了什么处理（mask、差分、log10、奇异值等）。
- **为什么能体现正确性**：为什么这个量能作为 `original` vs `hybrid-svd` 的正确性证据（物理意义、是否 gauge-invariant、对数值误差敏感性等）。

### 1) `H2_diag`（二体对角哈密顿量）
用于确认两种方法最终的对角化结果一致（核心物理量）。

- 图：`H2_diag_scatter.png`
- 数据：`H2_diag_scatter_data.npz`
- 如何计算/比较：
  - 从两边的 HDF5 中读取 `H2_diag`，并展平成一维向量。
  - 仅保留双方都为 finite 的元素（`mask = isfinite(a) & isfinite(b)`）。
  - 画散点图：横轴 original，纵轴 hybrid-svd，并画参考线 \(y=x\)。
  - `*_data.npz` 中保存了：
    - `a`/`b`：实际参与绘图的 filtered 数据
    - `lo`/`hi`：参考线绘制范围
- 数值指标（hybrid - original）：
  - RMSE `4.83e-08`
  - max_abs `3.07e-07`
  - 结论：**几乎完全一致**
- 为什么能体现正确性：
  - `H2_diag` 是流方程演化后得到的“有效对角哈密顿量（2-body 部分）”的直接输出；如果两种方法在 checkpoint/压缩上有系统性偏差，通常会首先反映在这里。
  - 该量不涉及 LIOM 的 gauge 自由度问题，比较是直接且物理明确的。

### 2) `Hint`（四体相互作用张量）
用于确认相互作用项在流动后的一致性（对压缩/数值误差较敏感）。

- 图：`Hint_log10_absdiff_hist.png`
- 数据：`Hint_log10_absdiff_hist_data.npz`
- 如何计算/比较：
  - 从两边 HDF5 读取 `Hint`，展平成一维向量 `a,b`。
  - 对 finite 元素做差：`diff = a - b`。
  - 画直方图的变量为：
    \[
    x = \log_{10}(|diff| + 10^{-18})
    \]
  - `*_data.npz` 中保存了：
    - `a`/`b`：参与比较的 filtered 原始数组
    - `diff`：差分
    - `log10_absdiff`：直方图输入 `x`
- 数值指标（hybrid - original）：
  - RMSE `1.52e-06`
  - max_abs `3.30e-05`
  - 结论：**整体一致**；直方图主体集中在很小的 `|Δ|`，尾部少量点更大。
  - 备注：`max_rel` 可能被“分母接近 0”的元素放大，因此相对误差不如绝对误差/分布直观。
- 为什么能体现正确性：
  - `Hint`（4-body 张量）比 `H2_diag` 更“高阶”和更敏感：如果 hybrid 的 checkpoint/压缩在反向重构时引入误差，往往会优先体现在 `Hint` 这种高维对象上。
  - 因此它是一个更严格的数值一致性检验；主体集中在极小 `|Δ|` 说明整体结构一致。

### 3) `trunc_err`（截断误差）
用于确认两边截断误差统计一致（如果这里差很多，通常意味着流程/实现不一致）。

- 图：`trunc_err_compare.png`
- 数据：`trunc_err_compare_data.npz`
- 数值指标：完全一致（RMSE=0, max_abs=0）
- 如何计算/比较：
  - 读取两边 `trunc_err`（脚本里按一维数组处理）。
  - 画线图时取共同长度 `n_common = min(len(a), len(b))` 并对齐前 `n_common` 项。
  - `*_data.npz` 中保存了 `a`/`b` 以及 `a_common`/`b_common`。
- 为什么能体现正确性：
  - 截断误差是 flow 过程中关于截断/丢弃项的统计量；如果两条实现路径在截断规则、数值稳定性或保存/重构流程上出现差异，`trunc_err` 常会出现系统偏离。
  - 这里完全一致是一个很强的“实现一致性”信号。

### 4) `dl_list`（flow 步长/步数轨迹）
用于观察两边的“收敛路径/步数”差异（并不要求长度一致）。

- 图：`dl_list_compare.png`
- 数据：`dl_list_compare_data.npz`
- 现象：original 长度 `4625`，hybrid-svd 长度 `5000`（shape mismatch）
- 解释：两边可能在不同 step 达到 cutoff，或 hybrid 达到上限步数；这不必然影响最终 `H2_diag/Hint` 的一致性结论。
- 如何计算/比较：
  - 读取两边 `dl_list`（每一步的步长或步长序列）。
  - 画线图时同样取 `n_common = min(len(a), len(b))` 并仅绘制前 `n_common` 项用于直观对比。
  - `*_data.npz` 中保存了 `a`/`b`（全长）与 `a_common`/`b_common`（对齐截断后）。
- 为什么能体现正确性：
  - `dl_list` 不是“物理输出量”，它是数值积分/步进策略的轨迹记录；两种方法即使得到相同终态，也可能经过不同的步数或步长分布。
  - 该图主要用于解释“为什么某些对象（尤其是 LIOM）会更敏感”：因为路径不同会放大/改变弱分量误差的累积方式。

### 5) `liom2`（二体 LIOM）
LIOM 的“具体矩阵元素”并不是唯一的：符号翻转、排序交换、（近）简并子空间内的正交旋转都会导致逐元素差分看起来很大。

因此这里采用 **gauge-invariant** 的比较方式：将 `liom2` 视为矩阵后比较其**奇异值谱**（singular values）。

- 图（物理不变）：`liom2_svals_compare.png`
- 数据：`liom2_svals_compare_data.npz`
- 图（物理不变差分直方图）：`liom2_log10_absdiff_hist.png`
- 数据：`liom2_log10_absdiff_hist_data.npz`

解读方式：

- 若奇异值谱整体对齐，说明 `liom2` 的“强成分/主结构”一致；
- 尾部小奇异值差异更明显通常表示：压缩/数值路径差异主要影响“弱分量/细节”，不一定意味着物理量（如 `H2_diag/Hint`）不一致。

> 注意：`summary.json` 中的 `liom2` 数值指标仍是逐元素 RMSE（不 gauge-invariant），因此可能显得偏大；应以 `liom2_svals_*` 两张图为主。
- 如何计算/比较（gauge-invariant）：
  - 从 HDF5 读取 `liom2`。
  - 若其形状本身是方阵 `(n,n)`，直接视为矩阵；若是一维但总长度是完全平方数，则 reshape 成 `(n,n)`。
  - 对矩阵做 SVD，取奇异值向量 \(\sigma\) 并按降序排序。
  - `liom2_svals_compare.png`：画 `sigma_original` vs `sigma_hybrid`（y 轴用 log）。
  - `liom2_log10_absdiff_hist.png`：对奇异值做差后画
    \[
    \log_{10}(|\Delta\sigma| + 10^{-18})
    \]
  - `*_data.npz` 中保存了 `svals_original` / `svals_hybrid_svd`（以及差分/直方图输入见对应文件）。
- 为什么能体现正确性：
  - LIOM 的“具体表示”不唯一（gauge 自由度），逐元素差分会把“符号/子空间旋转”误判成错误。
  - 奇异值谱对左右正交/酉变换不变，因此是更接近“物理指纹”的比较方式：它衡量算符整体强度在不同模式上的分解结构是否一致。
  - 尾部差异通常反映压缩与数值路径对弱分量的影响（更像“精度差异”而非“物理错误”）。

### 6) `liom4`（四体 LIOM）
目前仍使用逐元素差分直方图（更严格但不完全 gauge-invariant）。

- 图：`liom4_log10_absdiff_hist.png`
- 数据：`liom4_log10_absdiff_hist_data.npz`
- 如何计算/比较：
  - 读取 `liom4` 并展平，对 finite 元素做差。
  - 同 `Hint` 一样画 `log10(|diff| + 1e-18)` 的直方图，并在 `*_data.npz` 里保存 `a/b/diff/log10_absdiff`。
- 为什么能体现正确性：
  - `liom4` 是更高阶对象，对数值误差/压缩更敏感；逐元素直方图是严格的数值检验。
  - 但它也可能受某些 gauge/基底选择影响（比 `liom2` 更复杂），因此该图更适合作为“误差形态诊断”，不应单独作为是否正确的唯一依据。

---

## 总体结论（本次 run）

- **核心物理结果一致性很好**：`H2_diag` 基本完全重合，`Hint` 差异整体很小，`trunc_err` 完全一致。
- **收敛路径不同是可接受的**：`dl_list` 长度不同主要反映两边达到 cutoff 的步数/步长策略不同。
- **LIOM2 用物理不变指标后更可解释**：用奇异值谱可以避开 LIOM gauge 自由度带来的“假差异”，剩下差异更可能来自压缩/数值路径对弱分量的影响。

---

## 如何复现/加载每张图的数据

每张图旁边都有 `*_data.npz`，可用：

```python
import numpy as np
d = np.load("H2_diag_scatter_data.npz")
print(d.files)
```

