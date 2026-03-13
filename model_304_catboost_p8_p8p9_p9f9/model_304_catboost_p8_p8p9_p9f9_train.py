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
from catboost import CatBoostClassifier
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")

# ====================
# 1. 模型与参数配置
# ====================

MODEL_DEFINITIONS = {
    "p8":      {"cols": ["p8"],             "objective": "multi:softprob",    "num_class": 4, "is_binary": False},
    "p8p9":    {"cols": ["p8", "p9"],       "objective": "binary:logistic",   "num_class": None, "is_binary": True},
    "p9f9":    {"cols": ["p9", "f9"],       "objective": "multi:softprob",    "num_class": 3, "is_binary": False}
}

HYPERPARAMS = {
    "p8": {
"window_size": 244,
"sigma": 78,
"actual_step": 17,
"bootstrap_type": "Bernoulli",
"iterations": 1100,
"depth": 3,
"learning_rate": 0.012019739905723439,
"l2_leaf_reg": 3.692892226513041,
"subsample": 0.8,
"border_count": 128,
"random_strength": 3.64,
"task_type": "GPU",
"verbose": 0
    },
    "p8p9": {
"window_size": 269,
"sigma": 72,
"actual_step": 21,
"bootstrap_type": "Bernoulli",
"iterations": 1000,
"depth": 6,
"learning_rate": 0.08006721102127229,
"l2_leaf_reg": 0.041468807521091146,
"subsample": 0.8,
"scale_pos_weight": 9,
"task_type": "GPU",
"verbose": 0
    },
    "p9f9": {
"window_size": 329,
"sigma": 83,
"actual_step": 33,
"bootstrap_type": "Bernoulli",
"iterations": 1200,
"depth": 9,
"learning_rate": 0.04763763565341727,
"l2_leaf_reg": 0.0010188569933461905,
"subsample": 0.8,
"task_type": "GPU",
"verbose": 0
    }
}

# ==========================================
# 2. 工具函数
# ==========================================

def setup_logger(output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("CatBoostTrain")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(Path(output_dir) / "train_catboost_clean.log", mode='w', encoding='utf-8')
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

def get_catboost_loss(objective):
    if "binary" in objective: return "Logloss"
    if "multi" in objective: return "MultiClass"
    return "Logloss"

def load_dataset(path, anno_path, label_map, other_id=None):
    df = pd.read_csv(path)
    if "8p" in df.columns:
        df = df.rename(columns={"time": "date", "8p": "p8", "9p": "p9", "9f": "f9"})
    
    df = df.ffill()
    df['date'] = pd.to_datetime(df['date'].astype(str).apply(lambda x: x.split(".")[0]))
    df['day'] = df['date'].dt.date.apply(lambda x: x.replace(year=2025))
    
    annodata = pd.read_excel(anno_path)
    annodata['time'] = pd.to_datetime(annodata['time']).dt.date
    annodata['typea'] = annodata['type'].map(label_map)
    
    # --- 关键修改：将其他类清洗为 0 (背景) ---
    if other_id is not None:
        mask = annodata['typea'] == other_id
        if mask.any():
            annodata.loc[mask, 'typea'] = 0
            
    return df, annodata

def get_labeled_raw_data(df, annodata, model_type):
    cols = MODEL_DEFINITIONS[model_type]["cols"]
    days = sorted(list(set(df["day"])))
    data_list = []
    
    for t in days:
        sub = df[df['day'] == t][cols].copy()
        sub['anno'] = 0
        sub.reset_index(drop=True, inplace=True)
        
        day_annos = annodata[annodata['time'] == t]
        for _, row in day_annos.iterrows():
            s, e, tp = int(row['start']), int(row['end']), row['typea']
            s, e = max(0, s), min(len(sub) - 1, e)
            val = 0
            
            # 标签映射逻辑 (已移除 p_other)
            if model_type == "p8p9":
                if tp == 1: val = 1
            elif model_type == "p9f9":
                if tp == 2: val = 1
                elif tp == 6: val = 2
            elif model_type == "p8":
                if tp in [3, 4, 5]: val = tp - 2
            
            if val != 0 and e >= s: 
                sub.loc[s:e, 'anno'] = val
        data_list.append(sub)
    
    if not data_list: return pd.DataFrame()
    full_df = pd.concat(data_list, axis=0).reset_index(drop=True)
    full_df['anno'] = full_df['anno'].astype(int)
    return full_df

def build_features(raw_df, feature_cols, sigma, window_size, step_size):
    if raw_df.empty: return np.array([]), np.array([])
    df_temp = pd.DataFrame()
    for col in feature_cols:
        if sigma > 0: df_temp[col] = gaussian_filter(raw_df[col], sigma=sigma)
        else: df_temp[col] = raw_df[col]
            
    if len(raw_df) < window_size: return np.array([]), np.array([])

    label = raw_df['anno'].values[window_size-1:]
    raw_vals = df_temp[feature_cols].values
    num_feat = len(feature_cols)
    
    flattened = raw_vals.reshape(-1)
    real_window_len = window_size * num_feat
    slice_step = num_feat * step_size
    
    X = sliding_window_view(flattened, window_shape=real_window_len)[::slice_step]
    y = label[::step_size]
    
    min_len = min(len(X), len(y))
    return X[:min_len], y[:min_len]

# ==========================================
# 3. 主程序
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="../data/data3.csv")
    parser.add_argument("--anno_path", type=str, default="../data/anno_data9.0.xlsx")
    parser.add_argument("--label_map_path", type=str, default="../data/label.json")
    parser.add_argument("--output_dir", type=str, default="./saved_models_catboost_clean")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    args, unknown = parser.parse_known_args()

    logger = setup_logger(args.output_dir)
    
    with open(args.label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    
    # 识别 Other ID
    other_id = label_map.get("其他", None)
    if other_id:
        logger.info(f"Detected 'Other' Class ID: {other_id}. Will treat as Background (0).")

    logger.info("Loading Data...")
    df_train, annodata = load_dataset(args.train_data, args.anno_path, label_map, other_id)

    final_configs = {} 
    
    # 只遍历剩下的 3 个模型
    for model_name, def_cfg in MODEL_DEFINITIONS.items():
        logger.info(f"--- Training CatBoost Model: {model_name} ---")
        
        if model_name not in HYPERPARAMS:
            logger.error(f"Hyperparameters for {model_name} not found!")
            continue

        params = HYPERPARAMS[model_name].copy()
        win = int(params.pop("window_size"))
        sig = int(params.pop("sigma"))
        actual_step = int(params.pop("actual_step"))
        
        # 准备数据 (内部的 annodata 已经是清洗过的)
        raw_df = get_labeled_raw_data(df_train, annodata, model_name)
        
        if raw_df.empty or raw_df['anno'].max() == 0:
            logger.warning(f"No valid data for {model_name}, skipping.")
            continue
            
        X, y = build_features(raw_df, def_cfg["cols"], sig, win, actual_step)
        
        cb_params = params.copy()
        cb_params.update({
            "loss_function": get_catboost_loss(def_cfg["objective"]),
            "verbose": 100,
            "allow_writing_files": False,
            "task_type": "GPU" if args.device == "cuda" else "CPU",
        })
        
        logger.info(f"Start fitting...")
        clf = CatBoostClassifier(**cb_params)
        clf.fit(X, y)
        
        save_path = Path(args.output_dir) / f"{model_name}_catboost.joblib"
        joblib.dump(clf, save_path)
        logger.info(f"Model saved to {save_path}")

        final_configs[model_name] = {
            "model_path": str(save_path),
            "cols": def_cfg["cols"],
            "win": win,
            "sig": sig,
            "actual_step": actual_step,
            "model_type": "CatBoost"
        }

    config_path = Path(args.output_dir) / "model_configs.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(final_configs, f, indent=4)

    logger.info("All Training Complete.")

if __name__ == "__main__":
    main()
