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

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

# ==========================================
# 1. 单模型配置与 LSTM 超参数
# ==========================================

MODEL_DEFINITIONS = {
    "p_other": {
        "cols": ["p8", "p9", "f9"]
    }
}

# 保留有用的参数 (window_size, sigma, actual_step)，引入 LSTM 所需参数
HYPERPARAMS = {
    "p_other": {
        "window_size": int(100 * 1.0), 
        "sigma": int(15 * 1.0), 
        "actual_step": int(10 * 1.0),
        
        # LSTM 专属超参数
        "hidden_dim": 64,       # LSTM 隐藏层维度
        "num_layers": 2,        # LSTM 层数
        "learning_rate": 0.001, # 学习率
        "epochs": 30,           # 训练轮数
        "batch_size": 256       # 批大小
    }
}

# ==========================================
# 2. LSTM 模型定义
# ==========================================

class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(TimeSeriesLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # batch_first=True 表示输入数据的维度为 (batch, seq, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        out, (hn, cn) = self.lstm(x)
        # 取 LSTM 序列的最后一个时间步的输出用于分类
        out = self.fc(out[:, -1, :]) 
        return out

# ==========================================
# 3. 数据处理 (与原逻辑一致，修改了滑动窗口形状)
# ==========================================

def setup_logger(output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("TrainLogger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(Path(output_dir) / "training.log", mode='w', encoding='utf-8')
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

def load_dataset(path, anno_path, label_map):
    df = pd.read_csv(path)
    if "8p" in df.columns:
        df = df.rename(columns={"time": "date", "8p": "p8", "9p": "p9", "9f": "f9"})

    df = df.ffill()
    df['date'] = pd.to_datetime(df['date'].astype(str).apply(lambda x: x.split(".")[0]))
    df['day'] = df['date'].dt.date

    annodata = pd.read_excel(anno_path)
    annodata['time'] = pd.to_datetime(annodata['time']).dt.date
    annodata['typea'] = annodata['type'].map(label_map)

    return df, annodata

def build_contiguous_class_mapping(annodata):
    labels = sorted({int(x) for x in annodata['typea'].dropna().unique()})
    all_labels = [0] + [lb for lb in labels if lb != 0]
    old_to_new = {old: new for new, old in enumerate(all_labels)}
    new_to_old = {new: old for old, new in old_to_new.items()}
    num_class = len(all_labels)
    return old_to_new, new_to_old, num_class

def get_labeled_raw_data_all_classes(df, annodata, feature_cols, old_to_new):
    days = sorted(list(set(df["day"])))
    data_list = []

    for t in days:
        sub = df[df['day'] == t][feature_cols].copy()
        sub['anno'] = 0 
        sub.reset_index(drop=True, inplace=True)

        day_annos = annodata[annodata['time'] == t]
        for _, row in day_annos.iterrows():
            if pd.isna(row.get('typea')):
                continue
            s, e = int(row['start']), int(row['end'])
            s, e = max(0, s), min(len(sub) - 1, e)
            old_label = int(row['typea'])
            if old_label not in old_to_new:
                continue
            new_label = old_to_new[old_label]
            if e >= s:
                sub.loc[s:e, 'anno'] = new_label
        data_list.append(sub)

    if not data_list:
        return pd.DataFrame()

    full_df = pd.concat(data_list, axis=0).reset_index(drop=True)
    full_df['anno'] = full_df['anno'].astype(int)
    return full_df

def build_features_lstm(raw_df, feature_cols, sigma, window_size, step_size):
    """
    修改点：将展开(flatten)逻辑替换为生成 3D 序列数据用于 LSTM
    输出形状为 (N, window_size, num_feats)
    """
    df_temp = pd.DataFrame()
    for col in feature_cols:
        if sigma > 0:
            df_temp[col] = gaussian_filter(raw_df[col], sigma=sigma)
        else:
            df_temp[col] = raw_df[col]

    raw_vals = df_temp[feature_cols].values
    if len(raw_df) < window_size:
        return np.array([]), np.array([])

    label = raw_df['anno'].values[window_size - 1:]
    
    # 巧妙利用 sliding_window_view 在 axis=0 上切割，得到 (N, feature, window)
    view = sliding_window_view(raw_vals, window_shape=window_size, axis=0)
    # 转置为 LSTM 所需的 (N, window, feature)
    X = np.transpose(view, (0, 2, 1))[::step_size]
    y = label[::step_size]

    min_len = min(len(X), len(y))
    return X[:min_len], y[:min_len]

# ==========================================
# 4. 主程序
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="../data/data3.csv")
    parser.add_argument("--anno_path", type=str, default="../data/anno_data9.0_2021.xlsx")
    parser.add_argument("--label_map_path", type=str, default="../data/label.json")
    parser.add_argument("--output_dir", type=str, default="./saved_models")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    logger = setup_logger(args.output_dir)

    with open(args.label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    logger.info("Label Map Loaded.")

    logger.info("Loading Training Data...")
    df_train, annodata = load_dataset(args.train_data, args.anno_path, label_map)

    old_to_new, new_to_old, num_class = build_contiguous_class_mapping(annodata)
    logger.info(f"Detected num_class = {num_class} (including background=0).")

    final_configs = {}
    start_all_train = time.time()

    model_name = "p_other"
    def_cfg = MODEL_DEFINITIONS[model_name]
    logger.info(f"--- Training Model: {model_name} (ALL-CLASS) ---")

    params = HYPERPARAMS[model_name].copy()
    win = params["window_size"]
    sig = params["sigma"]
    actual_step = params["actual_step"]

    raw_df = get_labeled_raw_data_all_classes(
        df_train, annodata, def_cfg["cols"], old_to_new
    )

    if raw_df.empty:
        logger.error("No training data produced; abort.")
        return

    # 生成 LSTM 所需的 3D 张量数据
    X, y = build_features_lstm(raw_df, def_cfg["cols"], sig, win, actual_step)
    if len(X) == 0:
        logger.error("Not enough samples for windowing; abort.")
        return

    # 转换为 PyTorch Tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

    # 初始化模型
    device = torch.device(args.device)
    model = TimeSeriesLSTM(
        input_dim=len(def_cfg["cols"]), 
        hidden_dim=params["hidden_dim"], 
        num_layers=params["num_layers"], 
        num_classes=num_class
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    logger.info("Start training LSTM...")
    start_single_model = time.time()
    
    # LSTM 训练循环
    for epoch in range(params["epochs"]):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        acc = 100 * correct / total
        logger.info(f"Epoch [{epoch+1}/{params['epochs']}], Loss: {epoch_loss/len(dataloader):.4f}, Acc: {acc:.2f}%")

    duration = time.time() - start_single_model

    # 保存模型权重 (使用 PyTorch 格式)
    save_path = Path(args.output_dir) / f"{model_name}.pth"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path} (train_time={duration:.2f}s)")

    # 记录必要的架构参数，以便在推理时重新构建模型结构
    config_path = Path(args.output_dir) / "model_configs.json"
    final_configs[model_name] = {
        "model_path": str(save_path),
        "cols": def_cfg["cols"],
        "win": win,
        "sig": sig,
        "num_class": num_class,
        "hidden_dim": params["hidden_dim"],
        "num_layers": params["num_layers"],
        "old_to_new": old_to_new,
        "new_to_old": new_to_old
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(final_configs, f, indent=4, ensure_ascii=False)

    total_duration = time.time() - start_all_train
    logger.info(f"Total training process took: {total_duration:.2f} seconds")

if __name__ == "__main__":
    main()

