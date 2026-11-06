# PortPy 示例

每个示例都是一个 Jupyter notebook；其中大部分示例的纯 Python 版本位于
[`python_files/`](./python_files)。**需求**列列出了除 `pip install portpy`
之外还需要安装的内容。所有 notebook 默认使用 MOSEK；对于凸问题，任何 CVXPY
求解器都可以使用（PortPy 自带免费求解器 SCS 和 Clarabel，但它们比 MOSEK 或
Gurobi 慢得多；MOSEK 和 Gurobi 提供免费学术许可证）。

大多数 notebook 在首次运行时会从 Hugging Face 下载患者数据（使用专家选择的
射束时，每位患者约 2 GB）。参见主 README 的[数据](../README.md#data)部分。

> 说明：这是 `examples/README.md` 的中文版本，放在 `examples_zh/` 下。
> Notebook 中的代码单元保持原样，仅翻译说明性 Markdown 文本。对应的 Python
> 版本已复制到 `python_files/`。

## 从这里开始

| Notebook | 学习内容 | 需求 |
|---|---|---|
| [1_basic_tutorial](./1_basic_tutorial.ipynb) | 加载患者、创建 IMRT 计划、可视化与评估，以及保存和重新加载 | – |
| [structure_operations](./structure_operations.ipynb) | 结构的并集、交集、差集、膨胀和收缩 | – |
| [inf_matrix_down_sampling](./inf_matrix_down_sampling.ipynb) | 对 beamlet 和体素进行降采样以加速实验，以及这样做的代价 | – |
| [inf_matrix_sparsification](./inf_matrix_sparsification.ipynb) | 稀疏与完整影响矩阵；截断阈值 | – |

## IMRT 和 VMAT 计划设计

| Notebook | 学习内容 | 需求 |
|---|---|---|
| [vmat_scp_tutorial](./vmat_scp_tutorial.ipynb) | 使用列生成 + 序列凸规划进行 VMAT 计划设计；导出 DICOM-RT 计划 | – |
| [imrt_scp_tutorial](./imrt_scp_tutorial.ipynb) | 通过列生成 + SCP 设计每束多个孔径的 step-and-shoot IMRT | – |
| [dose_mimicking](./dose_mimicking.ipynb) | 优化计划以复现给定的参考剂量（剂量模仿） | Eclipse |

## 距离全局最优有多远？（混合整数规划参考）

| Notebook | 学习内容 | 需求 |
|---|---|---|
| [vmat_global_optimal](./vmat_global_optimal.ipynb) | 在降采样问题上与列生成和 SCP 对比的全局最优 VMAT 计划 | MIP 求解器 |
| [imrt_global_optimal](./imrt_global_optimal.ipynb) | 比较通量图 + 叶片序列、列生成、SCP 以及带可交付性约束的 MIP | Gurobi |
| [beam_orientation_global_optimal](./beam_orientation_global_optimal.ipynb) | IMRT 的全局最优射束角度 | MIP 求解器 |
| [dvh_constraint_global_optimal](./dvh_constraint_global_optimal.ipynb) | 满足剂量体积直方图约束的全局最优计划 | Gurobi |

## 基于 AI 的剂量预测（`portpy.ai`）

| Notebook | 学习内容 | 需求 |
|---|---|---|
| [beamlet_dose_prediction](./beamlet_dose_prediction.ipynb) | 任务 1：用 3D U-Net 预测各个 beamlet 的剂量，组装预测影响矩阵，并用它进行计划设计 | GPU，`portpy[ai]` |
| [imrt_dose_prediction](./imrt_dose_prediction.ipynb) | 任务 2：预测患者的三维剂量分布，并将其转化为可交付的 IMRT 计划 | GPU，`portpy[ai]` |
| [vmat_scp_dose_prediction](./vmat_scp_dose_prediction.ipynb) | VMAT 的任务 2：以预测剂量作为基于 SCP 的 VMAT 计划设计的参考 | GPU，`portpy[ai]` |

## 临床集成

| Notebook | 学习内容 | 需求 |
|---|---|---|
| [imrt_tps_import](./imrt_tps_import.ipynb) | 将 IMRT 计划进行叶片序列化并导出为 DICOM-RT 后导入 TPS；将最优通量导出到 Eclipse | TPS / Eclipse |
| [vmat_tps_import](./vmat_tps_import.ipynb) | 将 VMAT 计划导出为 DICOM-RT 并导入 TPS | TPS |
| [eclipse_photon_dose_calculation](./eclipse_photon_dose_calculation.ipynb) | 使用自己的患者数据：通过 Varian 的 [MAAS-DoseInfluenceMatrix](https://github.com/Varian-MedicalAffairsAppliedSolutions/MAAS-DoseInfluenceMatrix) 工具从 Eclipse 提取 beamlet、体素和影响矩阵，然后在 PortPy 中进行计划设计 | Eclipse |
| [3d_slicer_integration](./3d_slicer_integration.ipynb) | 在 Jupyter notebook 中使用 3D Slicer 查看 CT、结构和剂量 | 3D Slicer |
