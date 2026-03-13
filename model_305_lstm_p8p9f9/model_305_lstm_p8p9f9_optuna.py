# -*- coding: utf-8 -*-
import argparse
import json
import logging
import warnings
from pathlib import Path
import time
import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================
# 1. LSTM 模型定义
# ==========================================

class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.0):
        super(TimeSeriesLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # batch_first=True 表示输入数据的维度为 (batch, seq, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        out, (hn, cn) = self.lstm(x)
        # 取 LSTM 序列的最后一个时间步的输出用于分类
        out = self.fc(out[:, -1, :]) 
        return out


def r4(x):  # 保留 4 位
    return round(float(x), 4)


def setup_logger(output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("LSTMOptunaLogger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(Path(output_dir) / "lstm_optuna.log", mode="w", encoding="utf-8")
        ch = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# ==========================================
# 2. 数据处理函数
# ==========================================

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
    """
    逐天构建序列，输出 groups 向量（用于 GroupKFold，按天分组）
    """
    days = sorted(list(set(df["day"])))
    data_list = []
    group_list = []

    for gi, t in enumerate(days):
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
        group_list.append(np.full(len(sub), gi, dtype=int))

    if not data_list:
        return pd.DataFrame(), None

    full_df = pd.concat(data_list, axis=0).reset_index(drop=True)
    full_df['anno'] = full_df['anno'].astype(int)
    groups = np.concatenate(group_list, axis=0)
    return full_df, groups


def build_features_lstm(raw_df, groups, feature_cols, sigma, window_size, step_size):
    """
    生成 3D 序列数据用于 LSTM，输出形状为 (N, window_size, num_feats)
    同时对齐 groups
    """
    if raw_df.empty or groups is None:
        return np.array([]), np.array([]), np.array([])

    df_temp = pd.DataFrame()
    for col in feature_cols:
        if sigma > 0:
            df_temp[col] = gaussian_filter(raw_df[col].values, sigma=sigma)
        else:
            df_temp[col] = raw_df[col].values

    raw_vals = df_temp[feature_cols].values
    if len(raw_df) < window_size:
        return np.array([]), np.array([]), np.array([])

    label = raw_df['anno'].values[window_size - 1:]
    g_all = groups[window_size - 1:]
    
    # 滑动窗口视图
    view = sliding_window_view(raw_vals, window_shape=window_size, axis=0)
    # 转置为 LSTM 所需的 (N, window, feature)
    X = np.transpose(view, (0, 2, 1))[::step_size]
    y = label[::step_size]
    g = g_all[::step_size]

    min_len = min(len(X), len(y), len(g))
    return X[:min_len], y[:min_len], g[:min_len]


# ==========================================
# 3. Optuna 目标函数
# ==========================================

def lstm_objective(trial, raw_df, groups_raw, feature_cols, num_class, device):
    """
    LSTM 参数调优目标函数（优化内存使用）
    """
    # ---- A) 特征参数 ----（缩小范围以减少内存）
    win = trial.suggest_int("window_size", 30, 100)
    sig = trial.suggest_int("sigma", 0, 30)
    step_rate = trial.suggest_float("step_rate", 0.1, 0.5)
    actual_step = max(1, int(win * step_rate))

    # ---- B) LSTM 网络参数 ----（限制大小）
    hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
    num_layers = trial.suggest_int("num_layers", 1, 2)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)

    # ---- C) 训练参数 ----
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 64, 256)  # 减小batch size上限
    epochs = trial.suggest_int("epochs", 10, 30)

    # 构建数据
    X, y, g = build_features_lstm(raw_df, groups_raw, feature_cols, sig, win, actual_step)
    if len(y) < 200:
        return 0.0

    # 限制数据量以防止OOM
    max_samples = 5000
    if len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X = X[indices]
        y = y[indices]
        g = g[indices]

    # 转换为 PyTorch Tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # ---- D) GroupKFold 按天交叉验证 ----
    gkf = GroupKFold(n_splits=3)
    f1s = []

    try:
        for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=g)):
            # 确保验证集不为空
            if len(va_idx) == 0:
                continue
            
            # 创建模型
            model = TimeSeriesLSTM(
                input_dim=len(feature_cols), 
                hidden_dim=hidden_dim, 
                num_layers=num_layers, 
                num_classes=num_class,
                dropout=dropout
            ).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # 准备训练数据
            train_dataset = TensorDataset(X_tensor[tr_idx], y_tensor[tr_idx])
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # 准备验证数据 - 使用CPU验证减少GPU内存占用
            val_X = X_tensor[va_idx].clone()  # clone避免共享内存
            val_y_np = y_tensor[va_idx].numpy()

            # 训练
            model.train()
            for epoch in range(epochs):
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            # 验证 - 在CPU上进行
            model.eval()
            with torch.no_grad():
                # 分批验证以减少显存
                val_preds = []
                for i in range(0, len(val_X), 256):
                    batch_val = val_X[i:i+256].to(device)
                    outputs = model(batch_val)
                    _, predicted = torch.max(outputs, 1)
                    val_preds.append(predicted.cpu().numpy())
                val_preds = np.concatenate(val_preds)
                f1 = f1_score(val_y_np, val_preds, average="macro")
                f1s.append(f1)

            # 释放GPU内存
            del model, val_X
            torch.cuda.empty_cache() if device.type == "cuda" else None

        # 释放数据
        del X_tensor, y_tensor
        torch.cuda.empty_cache() if device.type == "cuda" else None

        return float(np.mean(f1s)) if f1s else 0.0
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            print(f"Trial failed due to OOM: {e}")
            torch.cuda.empty_cache() if device.type == "cuda" else None
            return 0.0
        raise
    except Exception as e:
        print(f"Trial failed with error: {e}")
        torch.cuda.empty_cache() if device.type == "cuda" else None
        return 0.0


# ==========================================
# 4. 主程序
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="../data/data3_hour_workday.csv")
    parser.add_argument("--anno_path", type=str, default="../data/anno_data9.0_2021.xlsx")
    parser.add_argument("--label_map_path", type=str, default="../data/label.json")
    parser.add_argument("--output_dir", type=str, default="./saved_models_lstm_optuna")
    parser.add_argument("--n_trials", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logger = setup_logger(args.output_dir)
    device = torch.device(args.device)
    logger.info(f"Using device: {args.device}")
    
    # 预先检查GPU内存并设置环境变量
    if device.type == "cuda":
        try:
            torch.cuda.set_per_process_memory_fraction(0.8)  # 最多使用80%显存
            import os
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        except:
            pass

    # 加载标签映射
    with open(args.label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    logger.info("Label Map Loaded.")

    # 加载训练数据
    logger.info("Loading Training Data...")
    df_train, annodata = load_dataset(args.train_data, args.anno_path, label_map)

    old_to_new, new_to_old, num_class = build_contiguous_class_mapping(annodata)
    logger.info(f"Detected num_class = {num_class} (including background=0).")

    # 特征列定义
    feature_cols = ["p8", "p9", "f9"]

    # 获取带分组标签的数据
    raw_df, groups_raw = get_labeled_raw_data_all_classes(
        df_train, annodata, feature_cols, old_to_new
    )

    if raw_df.empty:
        logger.error("No training data produced; abort.")
        return

    logger.info(f"Raw labeled sequence length: {len(raw_df)}. Starting Optuna...")

    # 开始 Optuna 调参
    tune_start = time.time()
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: lstm_objective(trial, raw_df, groups_raw, feature_cols, num_class, device),
        n_trials=args.n_trials
    )
    tune_time = time.time() - tune_start

    best_params = study.best_params
    best_score = study.best_value

    logger.info(f"Optuna finished. Time={tune_time:.2f}s, best_macroF1={best_score:.4f}")
    logger.info("Best params:")
    for k, v in best_params.items():
        logger.info(f"  {k}: {v}")

    # ---- 使用最优参数训练最终模型 ----
    best_win = int(best_params["window_size"])
    best_sig = int(best_params["sigma"])
    best_step_rate = float(best_params["step_rate"])
    best_actual_step = max(1, int(best_win * best_step_rate))
    best_hidden_dim = int(best_params["hidden_dim"])
    best_num_layers = int(best_params["num_layers"])
    best_dropout = float(best_params["dropout"])
    best_learning_rate = float(best_params["learning_rate"])
    best_batch_size = int(best_params["batch_size"])
    best_epochs = int(best_params["epochs"])

    # 构建最终训练数据
    X_final, y_final, _ = build_features_lstm(
        raw_df, groups_raw, feature_cols, best_sig, best_win, best_actual_step
    )

    # 转换为 PyTorch Tensors
    X_tensor = torch.tensor(X_final, dtype=torch.float32)
    y_tensor = torch.tensor(y_final, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=best_batch_size, shuffle=True)

    # 创建最终模型
    final_model = TimeSeriesLSTM(
        input_dim=len(feature_cols), 
        hidden_dim=best_hidden_dim, 
        num_layers=best_num_layers, 
        num_classes=num_class,
        dropout=best_dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_learning_rate)

    logger.info("Training final LSTM model with best params...")
    start_train = time.time()

    for epoch in range(best_epochs):
        final_model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = final_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        acc = 100 * correct / total
        logger.info(f"Epoch [{epoch+1}/{best_epochs}], Loss: {epoch_loss/len(dataloader):.4f}, Acc: {acc:.2f}%")

    train_time = time.time() - start_train

    # ---- 保存模型和配置 ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存模型权重
    model_path = out_dir / "p_other_lstm.pth"
    torch.save(final_model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    # 保存配置
    config = {
        "p_other": {
            "model_path": str(model_path),
            "cols": feature_cols,
            "win": best_win,
            "sig": best_sig,
            "actual_step": best_actual_step,
            "num_class": num_class,
            "hidden_dim": best_hidden_dim,
            "num_layers": best_num_layers,
            "dropout": r4(best_dropout),
            "learning_rate": r4(best_learning_rate),
            "batch_size": best_batch_size,
            "epochs": best_epochs,
            "old_to_new": old_to_new,
            "new_to_old": new_to_old,
            "best_params": {k: (r4(v) if isinstance(v, float) else int(v)) for k, v in best_params.items()},
            "cv_score_macro_f1": r4(best_score),
            "model_type": "TimeSeriesLSTM_Optuna_TPE"
        }
    }

    config_path = out_dir / "model_configs.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    logger.info(f"Config saved to {config_path}")

    logger.info(f"Total training time: {train_time:.2f}s")


if __name__ == "__main__":
    main()
