#!/usr/bin/env python
# DosePy -- OpenKBP experiment entry point
# DosePy -- OpenKBP 实验入口
#
# This file is the ONLY OpenKBP-specific Python file.  It contains OpenKBP data
# paths, parameters and model settings, and calls the reusable implementation
# that lives outside this folder.  The implementation files stay generic:
# 本文件是唯一与 OpenKBP 相关的 Python 文件。这里只放 OpenKBP 的数据路径、参数和
# 模型设置，并调用放在本文件夹之外的通用实现；那些实现文件本身保持通用，不含
# OpenKBP 专属逻辑：
#
#   data reading / CSV conversion / cache
#       DosePy/dosepy/data/openkbp_io.py
#   数据读取 / CSV 转换 / 缓存
#       DosePy/dosepy/data/openkbp_io.py
#   OpenKBP PyTorch dataset / 11-channel assembly
#       DosePy/dosepy/data/openkbp_dataset.py
#   OpenKBP PyTorch 数据集 / 11 通道组装
#       DosePy/dosepy/data/openkbp_dataset.py
#   training loop
#       DosePy/dosepy/train.py
#   训练主循环
#       DosePy/dosepy/train.py
#   single-patient inference
#       DosePy/dosepy/infer.py :: predict_openkbp_patient
#   单病例推理
#       DosePy/dosepy/infer.py :: predict_openkbp_patient
#   split-level testing (metrics + CSV/HTML outputs)
#       DosePy/dosepy/test.py :: test
#   整个划分的测试（指标 + CSV/HTML 输出）
#       DosePy/dosepy/test.py :: test
#   OpenKBP evaluation (overall + per-structure + DVH)
#       DosePy/dosepy/evaluation/openkbp.py :: evaluate_openkbp_dose
#   OpenKBP 评估（整体指标 + 分结构指标 + DVH）
#       DosePy/dosepy/evaluation/openkbp.py :: evaluate_openkbp_dose
#
# Usage examples:
#   python DosePy/OpenKBP/main.py --stage preprocess
#   python DosePy/OpenKBP/main.py --stage train
#   python DosePy/OpenKBP/main.py --stage infer --patient-id pt_241
#   python DosePy/OpenKBP/main.py --stage evaluate --patient-id pt_241
#   python DosePy/OpenKBP/main.py --stage evaluate       # whole test-pats split
#   python DosePy/OpenKBP/main.py --stage all
# 用法示例：
#   python DosePy/OpenKBP/main.py --stage preprocess        # 只做数据预处理/缓存
#   python DosePy/OpenKBP/main.py --stage train             # 训练
#   python DosePy/OpenKBP/main.py --stage infer --patient-id pt_241   # 单病例推理
#   python DosePy/OpenKBP/main.py --stage evaluate --patient-id pt_241 # 单病例评估
#   python DosePy/OpenKBP/main.py --stage evaluate          # 整个 test-pats 划分
#   python DosePy/OpenKBP/main.py --stage all               # 全流程
#
# Stages:
#   preprocess -> build .openkbp_cache from sparse CSVs
#   train      -> fit a model on train-pats, validate on validation-pats
#   infer      -> predict one patient, save DOSE_PRED/DOSE_GT/masks to .npz
#   evaluate   -> score one saved prediction, or run the whole test-pats split
#   all        -> preprocess + train + infer + evaluate one patient
# 阶段说明：
#   preprocess -> 把稀疏 CSV 转换成 .openkbp_cache 缓存
#   train      -> 在 train-pats 上训练，在 validation-pats 上验证
#   infer      -> 预测单个病人，把 DOSE_PRED/DOSE_GT/各结构 mask 存成 .npz
#   evaluate   -> 评估一个已保存的预测，或跑整个 test-pats 划分
#   all        -> 预处理 + 训练 + 单病例推理 + 单病例评估

import argparse
import sys
from pathlib import Path

# Make ``import dosepy`` work when this file is run directly from
# DosePy/OpenKBP/ without installing the package.
# 直接运行 DosePy/OpenKBP/ 下的本文件、且没有 pip install 本包时，
# 也让 ``import dosepy`` 能找到包。
DOSEPY_ROOT = Path(__file__).resolve().parents[1]
if str(DOSEPY_ROOT) not in sys.path:
    sys.path.insert(0, str(DOSEPY_ROOT))

import numpy as np

from dosepy.data.openkbp_io import (
    OPENKBP_CHANNEL_NAMES,
    OPENKBP_INPUT_NC,
    OPENKBP_PRESCRIPTION_GY,
    prepare_openkbp_cache,
)
from dosepy.evaluation.openkbp import evaluate_openkbp_dose
from dosepy.infer import predict_openkbp_patient
from dosepy.test import test
from dosepy.train import train

# =============================================================================
# OpenKBP experiment configuration
# OpenKBP 实验配置
# =============================================================================

# Paths. OpenKBP data and outputs stay inside the OpenKBP folder (gitignored).
# 路径。OpenKBP 数据与输出都放在 OpenKBP 文件夹内（已被 git 忽略）。
OPENKBP_DIR = Path(__file__).resolve().parent
DATA_ROOT = OPENKBP_DIR / "provided-data"       # 原始 OpenKBP CSV 数据根目录
WORK_ROOT = OPENKBP_DIR / "work"                # 实验输出根目录
CHECKPOINTS_DIR = WORK_ROOT / "checkpoints"     # 模型权重保存目录
RESULTS_DIR = WORK_ROOT / "results"             # 预测/评估结果目录

# Splits converted by the preprocess stage.
# 预处理阶段要转换的划分目录。
PREPROCESS_SPLITS = ("train-pats", "validation-pats", "test-pats")

# Experiment name and model.
# 实验名与模型设置。EXPECTED_DIR 为 <CHECKPOINTS_DIR>/<EXPERIMENT>。
EXPERIMENT = "openkbp_unet128"
NET_G = "unet_128"          # unet_128 | unet_256 | mednext | mednext_attn | mednext_duo   # 网络结构
NGF = 64                    # U-Net 基础通道数
NORM = "batch"              # U-Net 归一化方式：batch / instance / none
DOSE_LOSS = "mae"           # mae | mednext_hybrid | mae_moment   # 损失函数
N_EPOCHS = 100              # 训练轮数
BATCH_SIZE = 1              # 批大小（3D 数据通常为 1）
NUM_THREADS = 2             # DataLoader 线程数
LR = 1e-4                   # 学习率
GPU_IDS = [0]               # use [] for CPU   # 用 [] 表示 CPU
PLOT_LOSS = True            # 是否绘制 loss 曲线
PLOT_FREQ = 1               # 每 N 个 epoch 刷新一次 loss 曲线

# OpenKBP test / inference defaults
# OpenKBP 测试 / 推理默认值
TEST_PHASE = "test"         # 测试用的 phase 名（对应 test-pats）
TEST_PATIENT_ID = "pt_241"  # 默认单病例 id
TEST_EPOCH = "best_mae"     # checkpoint 选择器：best_mae / latest / 数字 epoch

# Model families other than the default U-Net: keep these in this file so the
# OpenKBP experiment parameters are all visible in one place.
# 默认 U-Net 之外模型的参数：集中放在这里，保证 OpenKBP 实验参数只在一个文件中。
NET_G_CFG = ""                                      # 生成器的额外构造参数（JSON 字符串，可留空）
MEDNEXT_MODEL_ID = "B"                              # MedNeXt 型号：S / B / M / L
MEDNEXT_KERNEL_SIZE = 3                             # MedNeXt 卷积核大小
MEDNEXT_N_CHANNELS = 32                             # mednext_attn / mednext_duo 的基础通道数
MEDNEXT_SITES = ("openkbp", "openkbp_aux")  # mednext_duo needs >=2 distinct names; OpenKBP uses the first; ignored by U-Net
# 上一行：mednext_duo 需要至少 2 个不同名字；OpenKBP 只用第一个；U-Net 会忽略该参数

# Input layout, derived from the OpenKBP channel table (do not hard-code 11).
# 输入布局：从 OpenKBP 通道表推导（不要手写 11）。
CHANNELS = OPENKBP_CHANNEL_NAMES    # (CT, 7 个 OAR, 3 个 PTV)
INPUT_NC = OPENKBP_INPUT_NC         # 输入通道数 = 11
OUTPUT_NC = 1                       # 输出通道数 = 1（剂量图）
PRESCRIPTION_GY = OPENKBP_PRESCRIPTION_GY   # 处方剂量 70 Gy


# =============================================================================
# [1] DATA PREPROCESSING
# [1] 数据预处理
# =============================================================================
# CALL: prepare_openkbp_cache(DATA_ROOT, ...)
# 调用: prepare_openkbp_cache(DATA_ROOT, ...)
# SOURCE: DosePy/dosepy/data/openkbp_io.py :: prepare_openkbp_cache
# 源码: DosePy/dosepy/data/openkbp_io.py :: prepare_openkbp_cache
# FORMAT: OpenKBP sparse CSV -> dense .npz cache; no PortPy.
# 格式: OpenKBP 稀疏 CSV -> 稠密 .npz 缓存；不依赖 PortPy。
def run_preprocess(force: bool = False):
    print("\n[1/4] OpenKBP data preprocessing")
    print("data root:", DATA_ROOT)
    # Build (or reuse) the per-patient .openkbp_cache files.
    # 生成（或复用）每个病人的 .openkbp_cache 缓存文件。
    counts = prepare_openkbp_cache(
        data_root=str(DATA_ROOT),
        splits=PREPROCESS_SPLITS,
        overwrite=force,
        verbose=True,
    )
    print("cached patients:", counts)
    return counts


# =============================================================================
# [2] TRAINING
# [2] 训练
# =============================================================================
# CALL: train(build_train_options())
# 调用: train(build_train_options())
# SOURCE: DosePy/dosepy/train.py :: train
# 源码: DosePy/dosepy/train.py :: train
# DATASET: DosePy/dosepy/data/openkbp_dataset.py :: OpenKBPDataset
# 数据集: DosePy/dosepy/data/openkbp_dataset.py :: OpenKBPDataset
#   - reads train-pats; validation reads validation-pats through val_phase
#   - dataset sets input_nc=11 automatically
#   - 训练读 train-pats；验证通过 val_phase 读 validation-pats
#   - 数据集会自动把 input_nc 设为 11
def build_train_options() -> dict:
    # All values mirror the module-level constants above.
    # 所有取值都对应上面的模块级常量。
    return {
        "dataroot": str(DATA_ROOT),
        "dataset_mode": "openkbp",          # 使用 OpenKBP dataset
        "model": "doseprediction3d",        # 训练用模型
        "name": EXPERIMENT,
        "checkpoints_dir": str(CHECKPOINTS_DIR),

        # Model / input layout (11 = CT + 7 OAR + 3 PTV)
        # 模型 / 输入布局（11 = CT + 7 个 OAR + 3 个 PTV）
        "input_nc": INPUT_NC,
        "output_nc": OUTPUT_NC,
        "netG": NET_G,
        "ngf": NGF,
        "norm": NORM,
        "dose_loss": DOSE_LOSS,
        "netG_cfg": NET_G_CFG,
        "mednext_model_id": MEDNEXT_MODEL_ID,
        "mednext_kernel_size": MEDNEXT_KERNEL_SIZE,
        "mednext_n_channels": MEDNEXT_N_CHANNELS,
        "mednext_sites": MEDNEXT_SITES,

        # Optimization
        # 优化相关
        "batch_size": BATCH_SIZE,
        "num_threads": NUM_THREADS,
        "gpu_ids": GPU_IDS,
        "n_epochs": N_EPOCHS,
        "n_epochs_decay": 0,                # 学习率线性衰减的 epoch 数
        "lr": LR,
        "lr_policy": "plateau",             # 用 val MAE 驱动 ReduceLROnPlateau

        # Validation / logging
        # 验证 / 日志
        "val_phase": "val",                 # 验证集 phase 名（对应 validation-pats）
        "plot_loss": PLOT_LOSS,             # 是否绘制 loss 曲线
        "plot_freq": PLOT_FREQ,             # loss 曲线刷新频率
        "save_epoch_freq": 10,              # 每 N 个 epoch 存一次数字 checkpoint
        "display_id": -1,                   # -1 关闭 visdom
        "no_html": True,                    # 不保存中间 HTML/nrrd
    }


def run_train():
    print("\n[2/4] OpenKBP training")
    options = build_train_options()
    print("experiment:", options["name"])
    print("input channels:", options["input_nc"])
    print("checkpoints:", options["checkpoints_dir"])
    # Training writes <checkpoints_dir>/<name>/ with:
    #   train_step_loss.csv, loss_curve.png, training_history.csv,
    #   latest_net_G.pth, <epoch>_net_G.pth, best_mae_net_G.pth
    # 训练会在 <checkpoints_dir>/<name>/ 下写入：
    #   train_step_loss.csv（逐 step loss）、loss_curve.png（loss 曲线）、
    #   training_history.csv（逐 epoch 指标）、latest/<epoch>/best_mae 权重。
    history = train(options)
    print("history:", history)
    return history


# =============================================================================
# [3] INFERENCE
# [3] 推理
# =============================================================================
# CALL: predict_openkbp_patient(...)
# 调用: predict_openkbp_patient(...)
# SOURCE: DosePy/dosepy/infer.py :: predict_openkbp_patient
# 源码: DosePy/dosepy/infer.py :: predict_openkbp_patient
# DATA: DosePy/dosepy/data/openkbp_io.py :: convert_openkbp_patient
# 数据: DosePy/dosepy/data/openkbp_io.py :: convert_openkbp_patient
# OUTPUT: results/<patient>_pred.npz
# 输出: results/<patient>_pred.npz
def run_infer(patient_id: str = TEST_PATIENT_ID, epoch: str = TEST_EPOCH):
    print("\n[3/4] OpenKBP inference")
    # Read one patient folder and reuse the trained 11-channel input layout.
    # 读取单个病人目录，并复用训练时的 11 通道输入布局。
    patient_dir = DATA_ROOT / "test-pats" / patient_id
    save_path = RESULTS_DIR / "{}_pred.npz".format(patient_id)
    result = predict_openkbp_patient(
        patient_dir=str(patient_dir),
        checkpoints_dir=str(CHECKPOINTS_DIR),
        name=EXPERIMENT,
        netG=NET_G,
        ngf=NGF,
        epoch=epoch,
        gpu_ids=GPU_IDS,
        norm=NORM,
        netG_cfg=NET_G_CFG,
        mednext_model_id=MEDNEXT_MODEL_ID,
        mednext_kernel_size=MEDNEXT_KERNEL_SIZE,
        mednext_n_channels=MEDNEXT_N_CHANNELS,
        mednext_sites=MEDNEXT_SITES,
        save_path=str(save_path),
    )
    # Saved arrays: DOSE_PRED, DOSE_GT, BODY, OAR, PTV, VOXEL_DIMS, PRESCRIPTION_GY
    # 保存的数组：DOSE_PRED, DOSE_GT, BODY, OAR, PTV, VOXEL_DIMS, PRESCRIPTION_GY
    print("saved prediction:", result.get("save_path", save_path))
    return result


# =============================================================================
# [4] EVALUATION
# [4] 评估
# =============================================================================
# Split-level test:
#   CALL: test(build_test_options())
#   SOURCE: DosePy/dosepy/test.py :: test
# 整个划分的测试：
#   调用: test(build_test_options())
#   源码: DosePy/dosepy/test.py :: test
# Single prediction:
#   CALL: evaluate_openkbp_dose(...)
#   SOURCE: DosePy/dosepy/evaluation/openkbp.py :: evaluate_openkbp_dose
# 单个已保存预测的评估：
#   调用: evaluate_openkbp_dose(...)
#   源码: DosePy/dosepy/evaluation/openkbp.py :: evaluate_openkbp_dose
def build_test_options(epoch: str = TEST_EPOCH) -> dict:
    # Test options must repeat the model settings used at training time, or the
    # checkpoint will not load into the rebuilt generator.
    # 测试参数必须与训练时一致，否则重建的生成器无法加载 checkpoint。
    return {
        "dataroot": str(DATA_ROOT),
        "dataset_mode": "openkbp",
        "model": "doseprediction3d",
        "name": EXPERIMENT,
        "checkpoints_dir": str(CHECKPOINTS_DIR),
        "results_dir": str(RESULTS_DIR),
        "phase": TEST_PHASE,                # 测试集 phase（对应 test-pats）
        "input_nc": INPUT_NC,
        "output_nc": OUTPUT_NC,
        "netG": NET_G,
        "ngf": NGF,
        "norm": NORM,
        "dose_loss": DOSE_LOSS,
        "netG_cfg": NET_G_CFG,
        "mednext_model_id": MEDNEXT_MODEL_ID,
        "mednext_kernel_size": MEDNEXT_KERNEL_SIZE,
        "mednext_n_channels": MEDNEXT_N_CHANNELS,
        "mednext_sites": MEDNEXT_SITES,
        "epoch": epoch,                     # 加载哪个 checkpoint
        "gpu_ids": GPU_IDS,
        "eval": True,                       # 推理前调用 model.eval()
    }


def run_evaluate(patient_id: str = None, epoch: str = TEST_EPOCH):
    print("\n[4/4] OpenKBP evaluation")
    if patient_id:
        # Single patient: read the npz written by run_infer().
        # 单病例：读取 run_infer() 写出的 npz。
        # Source of the saved tensors: run_infer() above.
        # 上面 run_infer() 保存的就是这些张量。
        pred_path = RESULTS_DIR / "{}_pred.npz".format(patient_id)
        if not pred_path.exists():
            raise FileNotFoundError(
                "{} not found; run --stage infer --patient-id {} first.".format(
                    pred_path, patient_id
                )
            )
        with np.load(pred_path, allow_pickle=False) as f:
            # Overall + per-structure + DVH metrics.
            # 整体指标 + 分结构指标 + DVH。
            evaluation = evaluate_openkbp_dose(
                pred_dose=f["DOSE_PRED"],
                gt_dose=f["DOSE_GT"],
                body=f["BODY"],
                oar=f["OAR"],
                ptv=f["PTV"],
                prescription_gy=float(f["PRESCRIPTION_GY"]),
            )
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stem = RESULTS_DIR / "{}_eval".format(patient_id)
        # Write the detail tables next to the prediction.
        # 把明细表写在预测结果旁边。
        evaluation["per_structure"].to_csv(str(stem) + "_per_structure.csv", index=False)
        evaluation["dvh"].to_csv(str(stem) + "_dvh.csv", index=False)
        print("overall metrics:", evaluation["overall"])
        print("per-structure CSV:", str(stem) + "_per_structure.csv")
        print("DVH CSV:", str(stem) + "_dvh.csv")
        return evaluation

    # Whole test split.  test() runs inference internally and writes
    # patient_metrics.csv plus the standard DosePy result folders.
    # 整个测试划分：test() 内部会跑推理，并写出 patient_metrics.csv
    # 以及 DosePy 的标准结果目录。
    options = build_test_options(epoch=epoch)
    summary = test(options)
    print("test summary:", summary)
    print("results:", options["results_dir"])
    return summary


# =============================================================================
# CLI
# 命令行入口
# =============================================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="OpenKBP experiment entry point for DosePy")
    # which stage to run: preprocess / train / infer / evaluate / all
    # 要运行的阶段：preprocess / train / infer / evaluate / all
    parser.add_argument(
        "--stage",
        choices=["preprocess", "train", "infer", "evaluate", "all"],
        default="train",
        help="which stage to run",
    )
    # OpenKBP patient folder name used by infer / evaluate
    # infer / evaluate 使用的 OpenKBP 病人目录名
    parser.add_argument(
        "--patient-id",
        default=None,
        help="OpenKBP patient folder name for infer/evaluate (default: {})".format(
            TEST_PATIENT_ID
        ),
    )
    # which checkpoint to load
    # 加载哪个 checkpoint
    parser.add_argument(
        "--epoch",
        default=TEST_EPOCH,
        help="checkpoint epoch selector (default: {})".format(TEST_EPOCH),
    )
    # rebuild existing .openkbp_cache files
    # 是否重建已存在的 .openkbp_cache 缓存
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild existing .openkbp_cache files during preprocess",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.stage == "preprocess":
        # [1] data preprocessing -> dosepy/data/openkbp_io.py
        # [1] 数据预处理 -> dosepy/data/openkbp_io.py
        run_preprocess(force=args.force)
    elif args.stage == "train":
        # [2] training -> dosepy/train.py
        # [2] 训练 -> dosepy/train.py
        run_train()
    elif args.stage == "infer":
        # [3] inference -> dosepy/infer.py
        # [3] 推理 -> dosepy/infer.py
        run_infer(patient_id=args.patient_id or TEST_PATIENT_ID, epoch=args.epoch)
    elif args.stage == "evaluate":
        # [4] evaluation -> dosepy/test.py or dosepy/evaluation/openkbp.py
        # [4] 评估 -> dosepy/test.py 或 dosepy/evaluation/openkbp.py
        # Without --patient-id this runs the whole test-pats split.
        # 不传 --patient-id 时跑整个 test-pats 划分。
        run_evaluate(patient_id=args.patient_id, epoch=args.epoch)
    elif args.stage == "all":
        # Full pipeline: preprocess -> train -> infer -> evaluate.
        # 全流程：预处理 -> 训练 -> 推理 -> 评估。
        run_preprocess(force=args.force)
        run_train()
        patient_id = args.patient_id or TEST_PATIENT_ID
        run_infer(patient_id=patient_id, epoch=args.epoch)
        run_evaluate(patient_id=patient_id, epoch=args.epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
