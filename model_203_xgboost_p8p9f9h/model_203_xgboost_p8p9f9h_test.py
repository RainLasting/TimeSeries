# -*- coding: utf-8 -*-
import argparse
import json
import logging
import warnings
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter
from sklearn.metrics import classification_report
from tqdm import tqdm

warnings.filterwarnings("ignore")


def setup_logger(output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("TestLogger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(ch)
    return logger


def load_dataset(path, anno_path, label_map):
    df = pd.read_csv(path)
    if "8p" in df.columns:
        df = df.rename(columns={"time": "date", "8p": "p8", "9p": "p9", "9f": "f9"})

    # 确保列存在且顺序一致
    df = df[["date", "p8", "p9", "f9", "hour"]]
    df = df.ffill()
    df["date"] = pd.to_datetime(df["date"].astype(str).apply(lambda x: x.split(".")[0]))
    df["day"] = df["date"].dt.date

    annodata = pd.read_excel(anno_path)
    annodata["time"] = pd.to_datetime(annodata["time"]).dt.date
    annodata["typea"] = annodata["type"].map(label_map)  # 原始 label_id（可能不连续）
    return df, annodata


def create_features_inference(df_segment, cols, sigma, window_size, step_size=1):
    """
    与训练 build_features 对齐：
    - 平滑
    - flatten
    - window_len = window_size * num_feat
    - stride = num_feat * step_size
    """
    smooth_data = []
    for c in cols:
        if sigma > 0:
            smooth_data.append(gaussian_filter(df_segment[c], sigma=sigma))
        else:
            smooth_data.append(df_segment[c].values)

    raw_values = np.column_stack(smooth_data)  # (N, num_feat)
    flattened = raw_values.reshape(-1)
    num_feat = len(cols)
    real_window_len = window_size * num_feat

    if len(flattened) < real_window_len:
        return None

    slice_step = num_feat * step_size
    feats = sliding_window_view(flattened, window_shape=real_window_len)[::slice_step]
    return feats


def adjust_predicts(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """后处理：区间填补（保留你的实现）"""
    predicted = predicted.copy()
    anomaly_state = False
    for i in range(len(actual)):
        flag = predicted[i]
        if actual[i] == predicted[i] != 0 and not anomaly_state:
            anomaly_state = True
            for j in range(i, -1, -1):
                if actual[j] == 0 or actual[j] != flag:
                    break
                if predicted[j] == 0:
                    predicted[j] = flag
            for j in range(i, len(actual)):
                if actual[j] == 0 or actual[j] != flag:
                    break
                if predicted[j] == 0:
                    predicted[j] = flag
        elif actual[i] == 0:
            anomaly_state = False
        if anomaly_state:
            predicted[i] = flag
    return predicted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, default="../data/data4_hour_workday.csv")
    parser.add_argument("--anno_path", type=str, default="../data/anno_data9.0_2021.xlsx")
    parser.add_argument("--label_map_path", type=str, default="../data/label.json")
    parser.add_argument("--model_dir", type=str, default="./saved_models_optuna")
    #parser.add_argument("--model_dir", type=str, default="./saved_models")
    parser.add_argument("--output_excel", type=str, default="test_report.xlsx",
                        help="Path to save the evaluation report excel")
    args = parser.parse_args()

    logger = setup_logger(args.model_dir)

    # 1) 加载配置
    config_path = Path(args.model_dir) / "model_configs.json"
    if not config_path.exists():
        logger.error("Run train.py first to generate configs.")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        model_configs = json.load(f)

    with open(args.label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    id_to_name = {v: k for k, v in label_map.items()}

    # 2) 只加载 all_with_hour 模型（单模型全类预测）
    if "all_with_hour" not in model_configs:
        logger.error("model_configs.json does not contain 'all_with_hour'. Please retrain with the new training code.")
        return

    cfg = model_configs["all_with_hour"]
    model_path = Path(cfg["model_path"])
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return

    model = joblib.load(model_path)

    cols = cfg["cols"]
    win = int(cfg["win"])
    sig = int(cfg["sig"])

    # new_to_old：把 contiguous 类别（0..K-1）映射回原始 label_id
    # json 读出来 key 可能是 str，需要转 int
    new_to_old = cfg.get("new_to_old", None)
    if new_to_old is None:
        logger.error("Config missing 'new_to_old'. Please retrain with the updated training code that saves mappings.")
        return
    new_to_old = {int(k): int(v) for k, v in new_to_old.items()}

    logger.info("Loading Test Data...")
    df_test, annodata = load_dataset(args.test_data, args.anno_path, label_map)
    days = sorted(list(set(df_test["day"])))

    full_preds, full_labels = [], []

    logger.info("Starting Inference...")
    inference_start = time.time()

    for t in tqdm(days, desc="Testing"):
        day_data = df_test[df_test["day"] == t].copy().reset_index(drop=True)
        N = len(day_data)

        # 构建 ground-truth（原始 label_id，背景=0）
        labels = np.zeros(N, dtype=int)
        for _, row in annodata[annodata["time"] == t].iterrows():
            if pd.isna(row.get("typea")):
                continue
            s, e, tp = int(row["start"]), int(row["end"]), int(row["typea"])
            s, e = max(0, s), min(N - 1, e)
            if e >= s:
                labels[s:e + 1] = tp

        # 特征
        X_feat = create_features_inference(day_data, cols, sig, win)

        if X_feat is None or len(X_feat) == 0:
            pred_seq = np.zeros(N, dtype=int)
        else:
            # contiguous 预测（0..K-1）
            pred_new = model.predict(X_feat).astype(int)

            # 映射回原始 label_id
            pred_old = np.array([new_to_old.get(int(i), 0) for i in pred_new], dtype=int)

            # padding：让序列长度对齐到 N（与你原来一致：前面补 0）
            pad = N - len(pred_old)
            if pad > 0:
                pred_seq = np.concatenate([np.zeros(pad, dtype=int), pred_old])
            else:
                pred_seq = pred_old[:N]

        full_preds.extend(pred_seq.tolist())
        full_labels.extend(labels.tolist())

    inference_end = time.time()
    inf_duration = inference_end - inference_start
    logger.info(f"Inference for {len(days)} days took: {inf_duration:.2f} seconds")
    logger.info(f"Average speed: {inf_duration/len(days):.4f} seconds/day")

    # 3) 评估
    full_labels = np.array(full_labels, dtype=int)
    full_preds = np.array(full_preds, dtype=int)

    # 后处理（保留你的逻辑）
    # preds_adj = adjust_predicts(full_labels, full_preds)
    preds_adj = full_preds
    # 只评估出现过的非 0 类
    unique_ids = sorted(list(set(full_labels) | set(preds_adj)))
    if 0 in unique_ids:
        unique_ids.remove(0)
    target_names = [id_to_name.get(i, f"Type {i}") for i in unique_ids]

    report_text = classification_report(
        full_labels, preds_adj,
        labels=unique_ids, target_names=target_names, digits=4
    )
    logger.info("\n" + "#" * 40 + "\nFINAL TEST REPORT\n" + "#" * 40 + "\n" + report_text)

    # 4) 保存 Excel
    report_dict = classification_report(
        full_labels, preds_adj,
        labels=unique_ids, target_names=target_names,
        output_dict=True
    )
    df_report = pd.DataFrame(report_dict).transpose().round(4)

    save_path = Path(args.output_excel)
    try:
        df_report.to_excel(save_path, float_format="%.4f")
        logger.info(f"Report successfully saved to Excel: {save_path.absolute()}")
    except Exception as e:
        logger.error(f"Failed to save Excel report: {e}")


if __name__ == "__main__":
    main()
