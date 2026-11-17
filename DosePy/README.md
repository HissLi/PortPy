# DosePy

**PortPy 全剂量预测（Full / Total Dose Prediction）代码整理版 —— anatomy-only / 7 通道**

DosePy 把 PortPy 中与 **total dose prediction** 相关的代码，从数据预处理、模型定义、参数、训练/测试、单病例推理，到评估与可视化，整理成一个独立的 Python 包，并配有一份中文全流程 notebook。本版本是 **anatomy-only 变体**：网络只接收 CT、OAR mask 和 PTV，不接收 BEAM 通道；因此不需要把 beam 作为模型输入，也不需要 `Beams/` 元数据或 influence matrix；这些只在评估/下游优化时使用。

> [`notebooks/01_full_dose_prediction_tutorial.ipynb`](notebooks/01_full_dose_prediction_tutorial.ipynb)

> 说明：本次整理**没有运行任何训练/推理代码，也没有执行任何 git 操作**。环境依赖、数据下载和 GPU 训练需要你在本地准备。

---

## 1. 任务定义

给定一个病人的 CT 和结构（OAR + PTV），预测其 **整个疗程的三维剂量分布**（单位 Gy），然后可以：

1. 与临床计划剂量比较（MAE、RMSE、R²、DVH）；
2. 用预测剂量做下游通量优化（dose mimicking / fluence optimization），得到可交付的计划；
3. 用 PortPy 的 scorecard 对计划打分。

DosePy 的网络输入为 **7 通道**，输出 **1 通道**剂量：

| 通道 | 内容 | 说明 |
|---|---|---|
| 0 | CT | HU 裁剪到 `[-1000, 3071]`，再缩放到 `[0, 1]` |
| 1–4 | OAR one-hot | 标签 1–4，分别对应各 site 的 4 个 OAR |
| 5 | 未使用 | `one_hot(..., 6)[..., 1:]` 的第 5 个通道；保留固定布局 |
| 6 | PTV | 0/1 |

空间尺寸固定为 `128 × 128 × 128`，由 PortPy 的 calculation box 裁剪后重采样得到。

> **与 PortPy 官方实现的区别**：PortPy 官方 total-dose baseline 是 8 通道（CT + BEAM + OAR + PTV），其中 BEAM 是 `A @ 1` 的 beam-sum dose prior。DosePy 当前版本按用户要求改成 **anatomy-only 7 通道**，因此 **不能加载 PortPy 官方 8 通道 checkpoint**；模型输入只需要 CT 和结构 mask，训练时另外需要 planner dose 作为监督标签。

支持两个 site：

| site | OAR（标签） | 默认处方 |
|---|---|---|
| `lung` | CORD(1), ESOPHAGUS(2), HEART(3), LUNGS_NOT_GTV(4) | 60 Gy（`Lung_2Gy_30Fx`） |
| `prostate` | BLADDER(1), RECTUM(2), FEMUR_L(3), FEMUR_R(4) | 70.2 Gy（`Prostate_26Fx`） |

---

## 2. 与 PortPy 的关系

DosePy **不是** PortPy 的替代品，而是从 `portpy.ai` 中抽出的 total-dose 代码，并改造成 anatomy-only 变体：

- **已迁移到 DosePy**：`portpy.ai.preprocess`（全剂量部分）、`portpy.ai.data`、`portpy.ai.models`（全剂量模型）、`portpy.ai.options`、`train.py`、`test.py`、`infer_runner.py`、`util`，以及评估/可视化所需的 DVH、scorecard 与画图代码。迁移后已移除 BEAM 通道，改为 7 通道 anatomy-only 输入。
- **仍依赖 PortPy**：`portpy.photon` 中的 `DataExplorer`（DosePy 用 `AnatomyOnlyDataExplorer`，不要求 `Beams/`）、`CT`、`Structures`、`ClinicalCriteria`、`Plan`、`Optimization`。预测本身不需要 `Beams` / `InfluenceMatrix`；只有下游 scorecard / 通量优化需要它们。
- **不在范围内**：PortPy.AI Task 1 的 beamlet dose prediction（`beamlet_*` 模型、`build_apred_batch.py`、`inf_matrix_predictor.py`、ray geometry 等），以及 VMAT/IMRT 优化教程。
- **本版本移除**：BEAM 输入通道、beam-sum 归一化、训练/推理对 `Beams` / `InfluenceMatrix` 的依赖，以及 PortPy 8 通道 checkpoint 兼容性。

完整的一一对应关系见 [`docs/CODE_MAP.md`](docs/CODE_MAP.md)。

---

## 3. 目录结构

```text
DosePy/
├── README.md
├── requirements.txt
├── pyproject.toml
├── notebooks/
│   └── 01_full_dose_prediction_tutorial.ipynb   # 中文全流程教程
├── docs/
│   └── CODE_MAP.md                              # PortPy -> DosePy 代码对照
└── dosepy/
    ├── __init__.py
    ├── preprocess/
    │   ├── site_config.py          # site / 结构 / 标签 / 处方配置
    │   ├── geometry.py             # calc-box 裁剪、128³ 重采样、拼接回 CT
    │   ├── metadata.py             # 只读 CT/structures/opt_voxels，不要求 Beams/
    │   ├── data_preprocess.py      # DICOM -> npz（训练/推理张量）
    │   ├── split_dataset.py        # 按 PortPy 冻结划分生成 train/val/test
    │   ├── predict.py              # 单病例预处理 + 推理 + 后处理
    │   └── postprocess.py          # 128³ 预测 -> CT 网格；存 nrrd
    ├── data/
    │   ├── base_dataset.py         # BaseDataset + 3D 增强 + 7 通道布局
    │   ├── dosepred3d_dataset.py   # 训练/验证/测试数据集
    │   ├── single_dataset.py       # 推理数据集（无 GT）
    │   ├── image_folder.py         # npz 文件枚举
    │   └── __init__.py             # dataset 工厂
    ├── models/
    │   ├── base_model.py
    │   ├── doseprediction3d_model.py  # 训练模型：loss + 验证指标
    │   ├── test_model.py              # 推理模型
    │   ├── networks3d.py              # define_G / 初始化 / scheduler
    │   ├── unet_generator3d.py        # unet_128 / unet_256
    │   ├── mednext_dose.py            # mednext
    │   ├── mednext_attention.py       # mednext_attn
    │   ├── mednext_duo.py             # mednext_duo（多 site）
    │   ├── losses3d.py                # MAE / weighted L1 / moment / gradient
    │   └── __init__.py                # model 工厂
    ├── options/
    │   ├── base_options.py            # parse(argv, overrides)
    │   ├── train_options.py
    │   ├── test_options.py
    │   └── __init__.py                # build_opt
    ├── train.py                       # 训练 + 验证 + checkpoint + CSV 历史
    ├── test.py                        # 测试/推理 + per-patient metrics
    ├── infer.py                       # InferenceRunner（模型只加载一次）
    ├── evaluation/
    │   ├── metrics.py                 # MAE/RMSE/R²/Pearson/nRMSE（整体 + 分结构）
    │   ├── dvh.py                     # DVH 曲线、D95/Dmean/V20 等
    │   └── scorecard.py               # DVH scorecard（PortPy scorecard 移植）
    ├── visualization/
    │   ├── dose_plots.py              # 预处理检查、切片/差分图
    │   ├── dvh_plots.py               # 多曲线 DVH、分结构指标柱状图
    │   ├── training_plots.py          # 训练曲线、per-patient 指标
    │   ├── report.py                  # 一键生成单病例评估报告
    │   └── portpy_plots.py            # PortPy plan 级可视化适配器
    ├── downstream/
    │   └── fluence_optimization.py    # 用预测剂量做通量优化
    ├── utils/
    │   ├── util.py
    │   ├── html.py
    │   └── visualizer.py
    └── configs/
        ├── scorecard/                 # 从 PortPy 复制的 lung/prostate scorecard
        └── data_splits/               # 从 PortPy 复制的冻结 train/val/test 划分
```

---

## 4. 安装

```bash
# 1) 安装 PortPy（如果还没装）
pip install portpy
# 或从 PortPy 源码目录：
# pip install -e /path/to/PortPy

# 2) 安装 DosePy
cd /path/to/PortPy/DosePy
pip install -r requirements.txt
pip install -e .
```

可选依赖：

```bash
# MedNeXt 系列模型（mednext / mednext_attn / mednext_duo）
pip install git+https://github.com/MIC-DKFZ/MedNeXt.git

# 下游通量优化建议用 MOSEK（学术免费）；没有 License 可用 SCS / CLARABEL
pip install Mosek

# 从 Hugging Face 下载 PortPy 数据
pip install "portpy[data]"
```

> **注意**：`unet_128` / `unet_256` 不需要 MedNeXt，是最小可运行配置。

---

## 5. 数据准备

DosePy 直接读取 PortPy 的数据目录。目录结构应类似：

```text
/path/to/portpy_data/data/
├── Lung_Patient_2/
│   ├── DicomFiles/
│   │   ├── rt_dose_echo_imrt.dcm      # 训练/验证需要（文件名由 technique_name 决定）
│   │   └── ...
│   ├── CT_MetaData.json
│   ├── StructureSet_MetaData.json
│   ├── OptimizationVoxels_MetaData.json
│   ├── PlannerBeams.json              # 仅下游 scorecard/优化需要
│   ├── Beams/                         # 仅下游 scorecard/优化需要
│   └── ...
├── Lung_Patient_3/
└── ...
```

下载示例：

```python
import portpy.photon as pp
pp.download_portpy_data(
    ["Lung_Patient_2", "Lung_Patient_3"],
    out="./portpy_data",
    # 只做 prediction 可用 beam_mode="none"；下游 scorecard/通量优化需要
    # influence matrix，此时用 "planner"。
    beam_mode="planner",
)
```

数据集划分使用 PortPy 的冻结配置（DosePy 随包附带副本 `dosepy/configs/data_splits/data_splits_default.json`）。**Task 2 的 lung validation/test 与 Task 1 是交换的**，不要混用；DosePy 的 `get_benchmark_split(site)` 默认读取 `total_dose_prediction`，是正确划分。

---

## 6. 快速开始

```python
import sys
sys.path.insert(0, "/path/to/PortPy/DosePy")

from dosepy.preprocess import (
    data_preprocess, split_preprocessed_dataset, predict_using_model,
)
from dosepy.train import train
from dosepy.test import test

DATA_DIR = "/path/to/portpy_data/data"
OUT_DIR = "/path/to/work/preprocessed_lung"

# 1) 预处理（生成 <Patient>.npz）
data_preprocess(
    in_dir=DATA_DIR,
    out_dir=OUT_DIR,
    site="lung",
    protocol_name="Lung_2Gy_30Fx",
    technique_name="imrt",
)

# 2) 按冻结 split 生成 train/val/test
split_preprocessed_dataset(OUT_DIR, site="lung")

# 3) 训练
history = train({
    "dataroot": OUT_DIR,
    "checkpoints_dir": "/path/to/work/checkpoints",
    "name": "dosepy_lung_unet128",
    "model": "doseprediction3d",
    "dataset_mode": "dosepred3d",
    "netG": "unet_128",
    "dose_loss": "mae",
    "input_nc": 7,
    "output_nc": 1,
    "batch_size": 1,
    "gpu_ids": [0],
    "n_epochs": 100,
})

# 4) 测试（有 GT，输出指标）
summary = test({
    "dataroot": OUT_DIR,
    "checkpoints_dir": "/path/to/work/checkpoints",
    "results_dir": "/path/to/work/results",
    "name": "dosepy_lung_unet128",
    "model": "doseprediction3d",
    "dataset_mode": "dosepred3d",
    "netG": "unet_128",
    "epoch": "best_mae",
    "eval": True,
})

# 5) 单病例预测
pred = predict_using_model(
    patient_id="Lung_Patient_94",
    in_dir=DATA_DIR,
    out_dir="/path/to/work/infer_input",
    model_name="dosepy_lung_unet128",
    checkpoints_dir="/path/to/work/checkpoints",
    results_dir="/path/to/work/results",
    netG="unet_128",
    site="lung",
    protocol_name="Lung_2Gy_30Fx",
    epoch="best_mae",
)
```

---

## 7. 每一步做什么

### 7.1 预处理

`dosepy.preprocess.data_preprocess` 对每个病人：

1. 读取 PortPy metadata，取 calculation box（`cal_box_xyz_start/end`）；
2. 把 CT、OAR、PTV、planner dose 裁剪到 calc box；
3. 重采样到 `128³`（CT/dose 线性插值，mask 最近邻）；
4. 生成 `OAR` 标签图与 `PTV` 二值图；
5. planner dose 裁剪到 `[0, 1.2 × 处方]`；
6. CT 裁剪/归一化；
7. 保存 `<Patient>.npz`（键：`CT, DOSE, OAR, PTV, BODY, PRESCRIPTION_GY`）。

整个过程不需要 beams、`Beams/` 元数据或 influence matrix；推理时使用同一流程，但不读取 planner dose，因此 npz 中没有 `DOSE`。

### 7.2 选择模型

在 `train_options["netG"]` 中指定：

| netG | 说明 | 依赖 |
|---|---|---|
| `unet_128` | 3D U-Net，7 次下采样；PortPy 历史默认，最轻量 | 无 |
| `unet_256` | 3D U-Net，8 次下采样；参数更多 | 无 |
| `mednext` | MedNeXt-B（GDP-HMM Top-1 风格） | `nnunet_mednext` |
| `mednext_attn` | MedNeXt + attention-gated skip（Top-2） | `nnunet_mednext` |
| `mednext_duo` | 共享 encoder + 每 site 一个 decoder（Top-3） | `nnunet_mednext` |

损失函数由 `dose_loss` 指定：

| dose_loss | 组成 |
|---|---|
| `mae` | 纯 L1 |
| `mae_moment` | L1 + ROI 剂量矩（1、2、10 阶） |
| `mednext_hybrid` | weighted L1 + body-masked weighted L1 + 剂量矩 + dual-gradient 边缘/平坦项 |

### 7.3 定义参数

所有参数既可以用 Python dict 传给 `train()` / `test()`，也能用命令行。`dosepy.options.build_opt` 可以在不启动训练的情况下检查最终参数：

```python
from dosepy.options import TrainOptions, build_opt
opt = build_opt(TrainOptions, {"dataroot": OUT_DIR, "n_epochs": 2, "gpu_ids": []})
print(opt)
```

关键参数（完整列表见 `dosepy/options/train_options.py`）：

| 参数 | 含义 |
|---|---|
| `dataroot` | 含 `train/val/test` 的预处理根目录 |
| `checkpoints_dir` / `name` | checkpoint 根目录 / 实验名 |
| `netG` | 生成器（见上表） |
| `dose_loss` | 损失（见上表） |
| `input_nc` / `output_nc` | 7 / 1（anatomy-only） |
| `batch_size` / `num_threads` | DataLoader |
| `gpu_ids` | `[0]` 用 GPU 0；`[]` 或 `-1` 用 CPU |
| `lr` / `lr_policy` | 初始学习率 / `linear`、`step`、`plateau`、`cosine` |
| `n_epochs` / `n_epochs_decay` | 训练轮数 |
| `lambda_L1` / `lambda_wL1` / `lambda_masked_wL1` / `lambda_moment` | 各损失权重 |
| `augment` | 是否做随机 3D 翻转/旋转/平移 |
| `val_phase` | 验证集子目录名（默认 `val`） |
| `epoch`（test） | `latest`、`best_mae` 或数字 epoch |

### 7.4 训练

`train(options)` 每个 epoch：

1. 训练一个 epoch；
2. 在 val split 上计算 loss、challenge-style MAE、R²、Pearson、nRMSE；
3. 用 val MAE 驱动 `ReduceLROnPlateau`（`lr_policy="plateau"`）；
4. 保存 `latest`、按 `save_epoch_freq` 保存数字 epoch、保存 `best_mae`；
5. 把每个 epoch 的指标写入 `<checkpoints_dir>/<name>/training_history.csv`；
6. 返回 history dict，可直接用 `plot_training_history(history)` 画图。

checkpoint 命名与 PortPy 兼容：`<epoch>_net_G.pth`，因此 PortPy 预训练权重可以直接用 `epoch="best_mae"` 加载。

### 7.5 测试

`test(options)` 遍历 `dataroot/test`：

- 每个病例保存预测剂量 `.../npz_images/<Patient>_CT2DOSE.nrrd`；
- 计算 loss、MAE、RMSE、R²、Pearson、nRMSE；
- 写 `.../patient_metrics.csv`（按 MAE 从差到好排序）；
- 生成 `.../index.html` HTML 索引（预测剂量以 `.nrrd` 保存；PNG 预览用 DosePy 的绘图函数生成）。

### 7.6 单病例预测

`predict_using_model(...)` 会：

1. 用与训练完全相同的预处理生成 `<out_dir>/infer/<patient>/<patient>.npz`；
2. 调 `test()`（或复用 `InferenceRunner`）加载 `best_mae` / `latest`；
3. 读回 `_CT2DOSE.nrrd`；
4. 重采样到裁剪后的 CT 尺寸；
5. 贴回完整 CT 网格（`origin/spacing/direction` 与 CT 一致）；
6. 返回 `(D, H, W)` 的预测剂量（Gy）。

预测本身**不建 BEAM**，也不需要 `Beams` / `InfluenceMatrix`。只有后面的 scorecard 和通量优化需要它们。

### 7.7 评估

```python
from dosepy.evaluation import dose_metrics, dose_metrics_by_structure, dvh_table

body = ct.ct_dict["ct_hu_3d"][0] > -500
overall = dose_metrics(pred, gt, mask=body, prescription_gy=60)

structure_masks = {
    name: structs.get_structure_mask_3d(name) > 0
    for name in ["PTV", "ESOPHAGUS", "HEART", "CORD", "LUNGS_NOT_GTV"]
}
per_structure = dose_metrics_by_structure(pred, gt, structure_masks, prescription_gy=60)
dvh_df = dvh_table(pred, gt, structure_masks, prescription_gy=60)
```

指标定义：

- **MAE / RMSE**：体素级剂量误差（Gy）；可选 `MAE(%Rx)` / `RMSE(%Rx)`；
- **R²**：`1 - SSE/SST`，衡量整体剂量分布解释度；
- **Pearson**：预测与 GT 的空间相关性；
- **nRMSE**：`RMSE / (GT_max - GT_min)`（PortPy 约定）；
- **DVH 指标**：`D95`、`D98`、`Dmean`、`V20Gy` 等；
- **scorecard**：PortPy 的临床评分卡，逐条 criterion 打分后求和。

### 7.8 可视化

- `plot_preprocessed_sample(npz)`：检查 7 通道预处理是否正确；
- `plot_slice_comparison(ct, {"Predicted": pred, "Reference": gt}, structures)`：CT + 预测 + GT + 差分切片；
- `plot_dvh_comparison({"Predicted": pred, "Reference": gt}, structures)`：多曲线 DVH；
- `plot_structure_metric_bars(metrics_df, metric="MAE")`：分结构指标柱状图；
- `plot_training_history(history)`：loss / MAE / R² 曲线；
- `save_evaluation_report(...)`：一键输出 PNG + CSV + summary；
- `plot_plan_dvh` / `plot_plan_2d_slice` / `plot_plan_fluence_3d`：PortPy plan 级图的适配器。

### 7.9 下游：用预测剂量做通量优化（可选）

```python
from dosepy.downstream import optimize_fluence_to_prediction

pred_dose_1d = inf_matrix.dose_3d_to_1d(pred)
result = optimize_fluence_to_prediction(my_plan, pred_dose_1d, solver="MOSEK")
opt_dose_1d = result["dose_1d"]
```

目标函数：

```text
min_x  ||A_PTV x - pred_PTV / nfx||²
     + w_OAR ||max(A_OAR x - pred_OAR / nfx, 0)||²
     + w_s · smoothness(x)
s.t.   x ≥ 0
```

---

## 8. 相对 PortPy 的主要改动

这些改动中，第 1 项是**有意的任务定义变化**（anatomy-only），其余是为了“能独立维护 + 能在 notebook 里稳定运行”：

1. **Anatomy-only 7 通道**：移除 BEAM 输入通道、beam-sum 归一化、训练/推理对 `Beams` / `InfluenceMatrix` 的依赖；新增 `AnatomyOnlyDataExplorer`，连 `Beams/` 元数据也不读取；`input_nc` 从 8 改为 7。**PortPy 官方 8 通道 checkpoint 不能加载**，需要重新训练。
2. **导入路径**：`portpy.ai.*` → `dosepy.*`。
3. **参数解析**：`BaseOptions.parse(argv=None, overrides=None)`，notebook 不再受 Jupyter `sys.argv` 干扰；未知参数名会直接报错。
4. **几何函数去重**：PortPy 在 `data_preprocess.py`、`predict_using_model.py`、`pred_dose_to_original_portpy.py` 三处重复了裁剪/重采样逻辑；DosePy 统一到 `preprocess/geometry.py`，并保留 calc-box 负索引 clamp。
5. **推理目录隔离**：PortPy 的 `out_dir/infer` 会累积所有病人的 npz，第二次推理会重复跑旧病人；DosePy 改为 `out_dir/infer/<patient_id>/`，每次只推理当前病人。
6. **验证指标**：新增 `DosePrediction3DModel.calculate_validation_metrics()`（MAE/RMSE/R²/corr/nRMSE），训练和测试使用同一套定义。
7. **body mask 兜底**：`mednext_hybrid` 在旧 npz 没有 `BODY` 时用 CT 阈值兜底，而不是直接 `None > 0.5` 报错。
8. **训练日志**：使用 `training_history.csv` 替代多个 `.xlsx`（不再隐式依赖 `openpyxl`），并返回 history dict。
9. **推理数据完整性**：`SingleDataset` 现在保留 `BODY` / `PRESCRIPTION_GY` / `SITE`（PortPy 版本在这三个字段上会丢失前两个）。
10. **推理 npz 不写 `DOSE=None`**：避免 object array 和 `allow_pickle` 问题；读取端仍兼容旧文件。
11. **范围裁剪**：删除 beamlet-dose（Task 1）模型、2D pix2pix 变换、GAN `ImagePool`、beamlet 专用参数；保留全剂量任务需要的全部代码。

---

## 9. 已知限制

- 预测/训练需要 PortPy 的元数据、CT、结构 mask；训练还需要 planner dose（DICOM RT dose）。不需要 beam 输入、`Beams/` 元数据或 influence matrix。
- 下游 scorecard / 通量优化仍需要 PortPy `Plan`，因此需要 influence matrix。
- `mednext*` 需要额外安装 `nnunet_mednext`；`unet_128` 无此依赖。
- 训练需要 GPU；CPU 只能跑很小的 smoke test。
- 下游通量优化默认 MOSEK；没有 License 时用 `solver="SCS"` 或 `"CLARABEL"`，速度较慢。
- `evaluation/scorecard.py` 的 DVH 插值仍调用 `portpy.photon.Evaluation`，因此需要 PortPy 的 `Plan`；纯预测指标（`metrics.py` / `dvh.py`）不依赖 PortPy。
- 目前只支持 `lung` / `prostate` 两个 site；新增 site 需要在 `preprocess/site_config.py` 增加配置，并准备对应 scorecard。
- anatomy-only 模型没有 BEAM 先验，通常比 PortPy 官方 8 通道 baseline 更难达到同样精度；这是去掉 beam 信息的预期代价。
- 旧的 PortPy 8 通道 `.npz` 仍可被本代码忽略 `BEAM` 读取，但旧的 8 通道 checkpoint 不能加载；需要重新训练 7 通道模型。

---

## 10. 许可证与引用

DosePy 迁移自 PortPy（Apache 2.0 + Commons Clause）。所有复制文件保留原始版权头；完整许可证见 [`LICENSE.txt`](LICENSE.txt)，使用时请遵守 PortPy 的非商业条款。

如果使用 PortPy / PortPy.AI，请引用：

```bibtex
@inproceedings{jhanwar2026portpyai,
  title     = {PortPy.AI: A Benchmark for AI and Optimization in Cancer Radiotherapy Planning},
  author    = {Jhanwar, Gourav and Tefagh, Mojtaba and Huang, Qijie and Golkarieh, Alireza and Hong, Linda and Pham, Hai and Taasti, Vicki Trier and Tuomaala, Seppo and Magliari, Anthony and Folkerts, Michael and Zhou, Ying and Yang, Jie and Nadeem, Saad and Zarepisheh, Masoud},
  year      = {2026}
}
```

PortPy 项目主页：<https://github.com/PortPy-Project/PortPy>
