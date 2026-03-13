# -*- coding: utf-8 -*-
import argparse
import json
import logging
import warnings
from pathlib import Path
import time
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter
from sklearn.metrics import classification_report
from tqdm import tqdm

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# 重新声明相同的 LSTM 模型类以便加载权重
class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.0):
        super(TimeSeriesLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out

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

    df = df[["date", "p8", "p9", "f9"]]
    df = df.ffill()
    df["date"] = pd.to_datetime(df["date"].astype(str).apply(lambda x: x.split(".")[0]))
    df["day"] = df["date"].dt.date

    annodata = pd.read_excel(anno_path)
    annodata["time"] = pd.to_datetime(annodata["time"]).dt.date
    annodata["typea"] = annodata["type"].map(label_map) 
    return df, annodata

def create_features_inference_lstm(df_segment, cols, sigma, window_size, step_size=1):
    """
    修改点：为 LSTM 生成 3D 推理数据 (N, window_size, feature)
    """
    smooth_data = []
    for c in cols:
        if sigma > 0:
            smooth_data.append(gaussian_filter(df_segment[c], sigma=sigma))
        else:
            smooth_data.append(df_segment[c].values)

    raw_values = np.column_stack(smooth_data)
    if len(raw_values) < window_size:
        return None

    view = sliding_window_view(raw_values, window_shape=window_size, axis=0)
    feats = np.transpose(view, (0, 2, 1))[::step_size]
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
    parser.add_argument("--model_dir", type=str, default="./saved_models_lstm_optuna")
    parser.add_argument("--output_excel", type=str, default="test_report.xlsx")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    logger = setup_logger(args.model_dir)
    device = torch.device(args.device)

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

    if "p_other" not in model_configs:
        logger.error("model_configs.json does not contain 'p_other'.")
        return

    cfg = model_configs["p_other"]
    model_path = Path(cfg["model_path"])
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return

    cols = cfg["cols"]
    win = int(cfg["win"])
    sig = int(cfg["sig"])
    
    # 构建并加载 LSTM 模型
    dropout = cfg.get("dropout", 0.0)  # 支持 optuna 调优后的 dropout 参数
    model = TimeSeriesLSTM(
        input_dim=len(cols), 
        hidden_dim=cfg["hidden_dim"], 
        num_layers=cfg["num_layers"], 
        num_classes=cfg["num_class"],
        dropout=dropout
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval() # 推理模式

    new_to_old = {int(k): int(v) for k, v in cfg.get("new_to_old", {}).items()}

    logger.info("Loading Test Data...")
    df_test, annodata = load_dataset(args.test_data, args.anno_path, label_map)
    days = sorted(list(set(df_test["day"])))

    full_preds, full_labels = [], []

    logger.info("Starting Inference...")
    inference_start = time.time()

    with torch.no_grad():
        for t in tqdm(days, desc="Testing"):
            day_data = df_test[df_test["day"] == t].copy().reset_index(drop=True)
            N = len(day_data)

            labels = np.zeros(N, dtype=int)
            for _, row in annodata[annodata["time"] == t].iterrows():
                if pd.isna(row.get("typea")):
                    continue
                s, e, tp = int(row["start"]), int(row["end"]), int(row["typea"])
                s, e = max(0, s), min(N - 1, e)
                if e >= s:
                    labels[s:e + 1] = tp

            X_feat = create_features_inference_lstm(day_data, cols, sig, win)

            if X_feat is None or len(X_feat) == 0:
                pred_seq = np.zeros(N, dtype=int)
            else:
                # 将 numpy 数组转为 tensor 并移至设备
                X_tensor = torch.tensor(X_feat, dtype=torch.float32).to(device)
                
                # --- 分批次推理逻辑 ---
                batch_size = 1024  # 你可以根据显存大小调整：512, 1024, 2048
                pred_new_list = []
                
                # 按 batch_size 遍历一整天的数据
                for i in range(0, len(X_tensor), batch_size):
                    batch_x = X_tensor[i : i + batch_size]
                    
                    # 经过模型前向传播
                    outputs = model(batch_x)
                    _, predicted = torch.max(outputs, 1)
                    
                    # 将预测结果存入列表
                    pred_new_list.append(predicted.cpu().numpy())
                
                # 拼接所有小 batch 的预测结果
                pred_new = np.concatenate(pred_new_list)
                
                # ================= 新补回的逻辑 =================
                # 1. 将预测出的连续类别（0,1,2...）映射回真实的 label_id
                pred_old = np.array([new_to_old.get(int(i), 0) for i in pred_new], dtype=int)

                # 2. padding：因为滑动窗口损失了最前面的 (window_size - 1) 个点，所以前面补 0 对齐长度 N
                pad = N - len(pred_old)
                if pad > 0:
                    pred_seq = np.concatenate([np.zeros(pad, dtype=int), pred_old])
                else:
                    pred_seq = pred_old[:N]
                # ===============================================

            full_preds.extend(pred_seq.tolist())
            full_labels.extend(labels.tolist())

    inference_end = time.time()
    inf_duration = inference_end - inference_start
    logger.info(f"Inference for {len(days)} days took: {inf_duration:.2f} seconds")

    # 3) 评估
    full_labels = np.array(full_labels, dtype=int)
    full_preds = np.array(full_preds, dtype=int)

    # 开启或关闭你原本的后处理
    # preds_adj = adjust_predicts(full_labels, full_preds)
    preds_adj = full_preds
    
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
