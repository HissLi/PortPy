# PortPy → DosePy 代码对照表

> **重要**：DosePy 有两条输入路径：**PortPy anatomy-only 7 通道**（`[CT, OAR-label-1..5, PTV]`）和 **OpenKBP 11 通道**（`CT + 7 OAR + 3 PTV`）。PortPy 路径移除了 BEAM 通道，因此与 PortPy 官方 8 通道 checkpoint 不兼容；OpenKBP 路径与 PortPy checkpoint 无关。本文件中的“保留流程”均指保留 CT/结构/dose 流程；模型输入/预测路径的 BEAM 部分已删除，评估/scorecard/下游优化仍会用到 `Beams` / `InfluenceMatrix`。

本文件记录 DosePy 中每个文件的来源，以及哪些 PortPy 代码**没有**迁移、为什么。目标：以后 PortPy 更新时，可以快速判断哪些改动需要同步。

## 1. 从 `portpy/ai` 迁移并重构

| PortPy 原文件 | DosePy 文件 | 处理方式 |
|---|---|---|
| `portpy/ai/preprocess/data_preprocess.py` | `dosepy/preprocess/data_preprocess.py` | 保留 CT/结构/dose 流程；**移除 BEAM 通道和 `pp.Beams`/`pp.InfluenceMatrix` 调用**；用 `AnatomyOnlyDataExplorer` 不要求 `Beams/` 元数据；几何函数拆到 `geometry.py`；site 配置拆到 `site_config.py` |
| `portpy/ai/preprocess/predict_using_model.py` | `dosepy/preprocess/predict.py` | 保留推理/后处理流程；**不建 BEAM、不调用 `pp.Beams`/`pp.InfluenceMatrix`**；用 `AnatomyOnlyDataExplorer` 不要求 `Beams/` 元数据；按病人隔离 inference 目录；新增 `save_nrrd` |
| `portpy/ai/preprocess/pred_dose_to_original_portpy.py` | `dosepy/preprocess/postprocess.py` | 去掉了硬编码 Windows 路径/病人列表，改成可复用函数（`resample_prediction_to_ct_grid` 等） |
| `portpy/photon/data_explorer.py::load_metadata` | `dosepy/preprocess/metadata.py` | 新增 `AnatomyOnlyDataExplorer`：structures / opt_voxels / CT 必需，`Beams/` 可选 |
| `portpy/ai/data/base_dataset.py` | `dosepy/data/base_dataset.py` | 只保留 3D 路径；删除 2D pix2pix transform（因此不再依赖 PIL/torchvision）；**输入布局改为 anatomy-only 7 通道** |
| `portpy/ai/data/dosepred3d_dataset.py` | `dosepy/data/dosepred3d_dataset.py` | 导入路径调整；删除未使用的导入；**不再读取/拼接 BEAM** |
| `portpy/ai/data/single_dataset.py` | `dosepy/data/single_dataset.py` | 导入路径调整；修复 `BODY` / `PRESCRIPTION_GY` 被丢弃的问题；**不再读取/拼接 BEAM** |
| `portpy/ai/data/image_folder.py` | `dosepy/data/image_folder.py` | 原样保留（仅改注释） |
| `portpy/ai/data/__init__.py` | `dosepy/data/__init__.py` | 去掉 `sys.path` hack；工厂逻辑保留 |
| `portpy/ai/models/base_model.py` | `dosepy/models/base_model.py` | 原样保留（相对导入已可用） |
| `portpy/ai/models/doseprediction3d_model.py` | `dosepy/models/doseprediction3d_model.py` | 导入路径调整；新增 `calculate_validation_metrics`；`mednext_hybrid` 增加 body mask 兜底；**结构通道切片从 `[:,2:]` 改为 `[:,1:]`** |
| `portpy/ai/models/test_model.py` | `dosepy/models/test_model.py` | 仅改导入 |
| `portpy/ai/models/networks3d.py` | `dosepy/models/networks3d.py` | 只保留全剂量生成器；删除 beamlet 分支；保留 `define_G` 签名与 `mednext_*` 参数 |
| `portpy/ai/models/unet_generator3d.py` | `dosepy/models/unet_generator3d.py` | 原样保留 |
| `portpy/ai/models/mednext_dose.py` | `dosepy/models/mednext_dose.py` | 原样保留 |
| `portpy/ai/models/mednext_attention.py` | `dosepy/models/mednext_attention.py` | 原样保留 |
| `portpy/ai/models/mednext_duo.py` | `dosepy/models/mednext_duo.py` | 原样保留 |
| `portpy/ai/models/losses3d.py` | `dosepy/models/losses3d.py` | 原样保留 |
| `portpy/ai/models/__init__.py` | `dosepy/models/__init__.py` | 去掉 `sys.path` hack；注册表逻辑保留 |
| `portpy/ai/options/base_options.py` | `dosepy/options/base_options.py` | 新增 `parse(argv=None, overrides=None)`；未知参数检查；`gpu_ids` 归一化；CPU 检查；**`input_nc` 默认 7** |
| `portpy/ai/options/train_options.py` | `dosepy/options/train_options.py` | 新增 `val_phase`；删除 beamlet 专用参数；保留全部全剂量参数 |
| `portpy/ai/options/test_options.py` | `dosepy/options/test_options.py` | 删除 beamlet `test_task`；保留全剂量参数 |
| `portpy/ai/options/__init__.py` | `dosepy/options/__init__.py` | 新增 `build_opt` |
| `portpy/ai/train.py` | `dosepy/train.py` | 重构：dict 参数、CSV 历史、返回 history、验证指标；逐 optimizer step 保存 `train_step_loss.csv`，默认每个 epoch 更新 `loss_curve.png`；训练逻辑与 checkpoint 命名保持一致 |
| `portpy/ai/test.py` | `dosepy/test.py` | 只保留全剂量测试；新增 per-patient CSV、返回 summary；保留 HTML/nrrd 输出 |
| `portpy/ai/infer_runner.py` | `dosepy/infer.py` | 使用 `build_opt`；复用 test 的保存逻辑；返回写出的文件路径 |
| `portpy/ai/util/util.py` | `dosepy/utils/util.py` | 原样保留 |
| `portpy/ai/util/html.py` | `dosepy/utils/html.py` | 原样保留 |
| `portpy/ai/util/visualizer.py` | `dosepy/utils/visualizer.py` | 原样保留 |
| `portpy/ai/statistics/compute_dvh_stats.py` | `dosepy/evaluation/dvh.py` | 从“写死路径的脚本”改成 numpy DVH 库函数 |
| `portpy/config_files/data_splits/data_splits_default.json` | `dosepy/configs/data_splits/data_splits_default.json` | 冻结的 Task 2 train/val/test 划分，随包发布 |
| `portpy/config_files/scorecard/*.json` | `dosepy/configs/scorecard/*.json` | lung / prostate scorecard |

## 2. 评估与可视化

| PortPy 代码 | DosePy 代码 | 说明 |
|---|---|---|
| `portpy/photon/utils/scorecard.py` | `dosepy/evaluation/scorecard.py` | 复制并加 `get_plan_score` 包装；修复 `plan.structs` → `plan.structures`；DVH 插值仍调用 PortPy `Evaluation` |
| `portpy/photon/evaluation.py::get_plan_score` | `dosepy/evaluation/scorecard.py::get_plan_score` | 等价入口；`DosePy` 版本默认 `return_score_df=True` |
| `portpy/photon/evaluation.py` 的 DVH 指标 | `dosepy/evaluation/dvh.py` | 提供不依赖 `Plan` 的 numpy 版本；数值定义更直接（未加权体素） |
| `portpy/photon/visualization.py::plot_dvh` | `dosepy/visualization/portpy_plots.py::plot_plan_dvh` | 适配器（保留 PortPy 的 plan-aware 渲染） |
| `portpy/photon/visualization.py::plot_2d_slice` | `dosepy/visualization/portpy_plots.py::plot_plan_2d_slice` | 适配器 |
| `portpy/photon/visualization.py::plot_fluence_3d` | `dosepy/visualization/portpy_plots.py::plot_plan_fluence_3d` | 适配器 |
| （无直接对应） | `dosepy/visualization/dose_plots.py` | 新增：预处理检查、CT/预测/GT/差分切片 |
| （无直接对应） | `dosepy/visualization/dvh_plots.py` | 新增：多曲线 DVH、分结构指标柱状图 |
| （无直接对应） | `dosepy/visualization/training_plots.py` | 新增：逐 step loss 日志 + 训练中实时 loss 曲线（`LossCurvePlotter`）、训练结束曲线、per-patient 指标 |
| （无直接对应） | `dosepy/visualization/report.py` | 新增：单病例报告（PNG + CSV + summary） |
| （无直接对应） | `dosepy/evaluation/metrics.py` | 新增：体素/分结构 MAE、RMSE、R²、Pearson、nRMSE |

## 2.1 OpenKBP 适配（非 PortPy 来源）

| 来源 / 文件 | DosePy 文件 | 说明 |
|---|---|---|
| `DosePy/OpenKBP/provided-data/{train,validation,test}-pats` | — | OpenKBP 原始 CSV 数据，按 split 直接使用 |
| OpenKBP 稀疏 CSV 格式 | `dosepy/data/openkbp_io.py` | 数据读取/归一化/缓存：`ct.csv`、`dose.csv`、`possible_dose_mask.csv`、结构 CSV -> dense arrays；`prepare_openkbp_cache` 批量生成 `.openkbp_cache` |
| OpenKBP 结构定义 | `dosepy/data/openkbp_io.py` | 7 个 OAR（Brainstem、SpinalCord、RightParotid、LeftParotid、Mandible、Esophagus、Larynx）+ 3 个 PTV（PTV56/63/70） |
| OpenKBP 输入组装 | `dosepy/data/openkbp_dataset.py` | `OpenKBPDataset`：11 通道 `[CT, 7 OAR, 3 PTV]`，`BODY` 单独返回；自动设置 `input_nc=11` |
| OpenKBP 单病例推理 | `dosepy/infer.py` | `predict_openkbp_patient`：构建单病人输入、加载 checkpoint、输出 pred/GT/masks npz |
| OpenKBP 评估 | `dosepy/evaluation/openkbp.py` | `evaluate_openkbp_dose`：整体指标 + per-structure 指标 + DVH |
| OpenKBP 实验入口 | `DosePy/OpenKBP/main.py` | 唯一 OpenKBP 专属 py：数据路径/参数/模型设置 + 分阶段调用；每步注释标源码位置 |

OpenKBP 分支不使用 `portpy.photon`，也不需要 `data_preprocess` / `split_preprocessed_dataset`；
`dataset_mode="openkbp"` 会自动设置 `input_nc=11` 并按 `train-pats` / `validation-pats` /
`test-pats` 映射 phase。

## 3. 下游优化

| PortPy 教程 | DosePy 代码 | 说明 |
|---|---|---|
| `examples/imrt_dose_prediction.ipynb` 第 4 节（fluence optimization） | `dosepy/downstream/fluence_optimization.py` | 把 notebook 里的 CVXPY 目标封装成 `optimize_fluence_to_prediction` |

## 4. 明确未迁移的代码

| 未迁移 | 原因 |
|---|---|
| `portpy/ai/preprocess/beamlet_preprocess_utils.py` | Task 1（beamlet dose prediction），不是全剂量预测 |
| `portpy/ai/preprocess/beamletdose_preprocess.py` | Task 1 |
| `portpy/ai/preprocess/build_apred_batch.py` | Task 1：构造预测的 influence matrix |
| `portpy/ai/inference/*`（`inf_matrix_predictor.py`、`ray_geometry.py`、`patient_grid_geometry.py`、`gpu_resample.py`） | Task 1：beamlet / ray patch 推理 |
| `portpy/ai/models/beamlet_*.py` | Task 1 的生成器 |
| `portpy/ai/models/beamletdose3d_model.py` | Task 1 的训练模型 |
| `portpy/ai/data/beamletdose3d_dataset.py` | Task 1 的数据集 |
| `portpy/ai/util/image_pool.py` | GAN 专用；全剂量模型没有判别器 |
| `portpy/ai/train_upkern.py` | UpKern 训练入口；UpKern 权重加载逻辑仍保留在 `DosePrediction3DModel.load_upkern_weights` |
| `portpy/ai/data/template_dataset.py` | 模板文件，无实际功能 |
| `portpy/photon/*` 本身 | 数据访问、剂量计算、优化、计划对象；DosePy 保留为依赖，不复制 |

## 5. 关键 API 对照

| PortPy | DosePy |
|---|---|
| `portpy.ai.preprocess.data_preprocess.data_preprocess` | `dosepy.preprocess.data_preprocess` |
| `portpy.ai.preprocess.predict_using_model.predict_using_model` | `dosepy.preprocess.predict.predict_using_model` |
| `portpy.ai.train.train` | `dosepy.train.train` |
| `portpy.ai.test.test` | `dosepy.test.test` |
| `portpy.ai.infer_runner.InferenceRunner` | `dosepy.infer.InferenceRunner` |
| `portpy.ai.models.networks3d.define_G` | `dosepy.models.networks3d.define_G` |
| `TrainOptions().parse()` | `TrainOptions().parse(argv=[], overrides={...})` 或 `build_opt(TrainOptions, {...})` |
| `pp.Evaluation.get_plan_score` | `dosepy.evaluation.get_plan_score`（实现同源） |
| `pp.Visualization.plot_dvh` | `dosepy.visualization.plot_plan_dvh`（适配器） |

## 6. 同步 PortPy 更新时的注意事项

1. **Anatomy-only 布局**：`[CT, OAR-label-1..5, PTV]` = 7 通道；BEAM 已移除。PortPy 官方 8 通道 checkpoint 不能加载，必须重新训练。
2. **通道索引**：`real_CT[:, 1:]` 是 6 个结构通道（4 个 OAR + 1 个空 + PTV），`real_CT[:, -1:]` 是 PTV；不要改回 `[:, 2:]`。
3. **归一化**：CT `(HU+1000)/4071`；dose clip 到 `1.2*Rx`。不再有 beam 归一化。
4. **checkpoint 命名不能变**：`<epoch>_net_G.pth`。
5. **几何裁剪约定不能变**：slice 轴 inclusive（`end+1`），y/x 轴 exclusive；calc-box 负索引 clamp。
6. 如果 PortPy 更新了 `mednext_*.py` 或 `unet_generator3d.py`，直接覆盖 DosePy 对应文件即可（它们没有 DosePy 改动）。
7. 如果 PortPy 更新了 `scorecard.py`，需要同步并保留 `plan.structures` 修复。
8. `scorecard` / `downstream` 仍使用 `Plan` / `InfluenceMatrix`，但这是评估/下游需求，不是预测输入。
