# -*- coding: utf-8 -*-
"""
train.py: 使用最优参数训练模型并保存
"""
import argparse
import json
import logging
import os
import warnings
from pathlib import Path
import time 
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")

# ==========================================
# 1. 参数配置
# ==========================================
# "win": 窗口大小
# "sig": 高斯平滑 sigma
# "step": 采样步长 (window_size * step_rate)
# "xgb": XGBoost 的参数字典
# ==========================================

BEST_PARAMS = {
    "p8": {
"win": 244,
"sig": 78,
"step": 17,
"xgb": {
"subsample": 0.8, 
"colsample_bytree": 0.8, 
"gamma": 3.64, 
"learning_rate": 0.012019739905723439, 
"reg_alpha": 0.013281081637581491, 
"reg_lambda": 3.692892226513041, 
"n_estimators": 1094, 
"max_depth": 3,

            "device": "cuda",
            "tree_method": "hist",
            "n_jobs": -1
        }
    },
    "p8p9": {
"win": 269,      
"sig": 72,       
"step": 21,      
        "xgb": {
"subsample": 0.8, 
"colsample_bytree": 0.8, 
"gamma": 4.6168000000000005, 
"learning_rate": 0.08006721102127229, 
"reg_alpha": 0.008604378874227247, 
"reg_lambda": 0.041468807521091146, 
"n_estimators": 1000, 
"max_depth": 6, 
"scale_pos_weight": 9,

            "device": "cuda",
            "tree_method": "hist",
            "n_jobs": -1
        }
    },
    "p9f9": {
"win": 329,      
"sig": 83,       
"step": 33,      
        "xgb": {
"subsample": 0.8, 
"colsample_bytree": 0.8, 
"gamma": 3.5469000000000004, 
"learning_rate": 0.04763763565341727, 
"reg_alpha": 0.02983254075991915, 
"reg_lambda": 0.0010188569933461905, 
"n_estimators": 1186, 
"max_depth": 9,
            
            "device": "cuda",
            "tree_method": "hist",
            "n_jobs": -1
        }
    }
}

MODEL_DEFINITIONS = {
    "p8":   {"cols": ["p8"],       "objective": "multi:softprob", "num_class": 4},
    "p8p9": {"cols": ["p8", "p9"], "objective": "binary:logistic", "num_class": None},
    "p9f9": {"cols": ["p9", "f9"], "objective": "multi:softprob", "num_class": 3}
}

def setup_logger(output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("TrainLogger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(Path(output_dir) / "train.log", mode='w', encoding='utf-8')
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

def load_dataset(path, anno_path, label_map, other_id):
    df = pd.read_csv(path)
    df = df[["time", "8p", "9p", "9f"]]
    df.columns = ["date", "p8", "p9", "f9"]
    df = df.ffill()
    df['date'] = pd.to_datetime(df['date'].astype(str).apply(lambda x: x.split(".")[0]))
    df['day'] = df['date'].dt.date.apply(lambda x: x.replace(year=2025))
    
    annodata = pd.read_excel(anno_path)
    annodata['time'] = pd.to_datetime(annodata['time']).dt.date
    annodata['typea'] = annodata['type'].map(label_map)
    
    # 清洗其他类 -> 0
    if other_id is not None:
        mask = annodata['typea'] == other_id
        if mask.any():
            annodata.loc[mask, 'typea'] = 0
            
    return df, annodata

def get_labeled_data(df, annodata, model_type):
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
    return pd.concat(data_list, axis=0).reset_index(drop=True)

def build_features(raw_df, feature_cols, sigma, window_size, step_size):
    df_temp = pd.DataFrame()
    for col in feature_cols:
        if sigma > 0: df_temp[col] = gaussian_filter(raw_df[col], sigma=sigma)
        else: df_temp[col] = raw_df[col]
            
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default="../data/data4.csv")
    parser.add_argument("--anno_path", default="../data/anno_data9.0.xlsx")
    parser.add_argument("--label_map", default="../data/label.json")
    parser.add_argument("--out_dir", default="./final_models")
    args = parser.parse_args()
    
    logger = setup_logger(args.out_dir)
    
    with open(args.label_map, 'r') as f:
        label_map = json.load(f)
    other_id = label_map.get("其他", 7)
    
    logger.info("Loading Training Data...")
    df_train, annodata = load_dataset(args.train_data, args.anno_path, label_map, other_id)
    
    metadata = {}

    total_start_time = time.time()

    for m_type in ["p8", "p8p9", "p9f9"]:
        logger.info(f"Training Model: {m_type}")
        model_start_time = time.time()
        # 1. 获取参数
        params = BEST_PARAMS[m_type]
        def_cfg = MODEL_DEFINITIONS[m_type]
        
        # 2. 准备数据
        train_raw = get_labeled_data(df_train, annodata, m_type)
        X, y = build_features(train_raw, def_cfg["cols"], params["sig"], params["win"], params["step"])
        
        # 3. 训练
        xgb_args = params["xgb"].copy()
        xgb_args["objective"] = def_cfg["objective"]
        if def_cfg["num_class"]:
            xgb_args["num_class"] = def_cfg["num_class"]
            
        clf = xgb.XGBClassifier(**xgb_args)
        clf.fit(X, y)

        model_end_time = time.time()
        model_duration = model_end_time - model_start_time
        logger.info(f"Finished {m_type} in {model_duration:.2f} seconds.")

        # 4. 保存模型
        model_path = Path(args.out_dir) / f"{m_type}.joblib"
        joblib.dump(clf, model_path)
        logger.info(f"Saved model to {model_path}")
        
        # 5. 记录元数据 (测试时必须用同样的 window 和 sigma)
        metadata[m_type] = {
            "cols": def_cfg["cols"],
            "win": params["win"],
            "sig": params["sig"],
            "path": str(model_path),
            "train_time": round(model_duration, 2)
        }
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    # 保存元数据文件
    with open(Path(args.out_dir) / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Training Complete. Total time: {total_duration:.2f} seconds.")

if __name__ == "__main__":
    main()