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

# ==========================================
# 1. 基础配置与工具
# ==========================================

def setup_logger(output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ExpertVotingTest")
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
    
    df = df.ffill()
    df['date'] = pd.to_datetime(df['date'].astype(str).apply(lambda x: x.split(".")[0]))
    df['day'] = df['date'].dt.date
    
    annodata = pd.read_excel(anno_path)
    annodata['time'] = pd.to_datetime(annodata['time']).dt.date
    annodata['typea'] = annodata['type'].map(label_map)
    return df, annodata

def create_features_inference(df_segment, cols, sigma, window_size):
    smooth_data = []
    for c in cols:
        if sigma > 0:
            smooth_data.append(gaussian_filter(df_segment[c], sigma=sigma))
        else:
            smooth_data.append(df_segment[c].values)
    
    raw_values = np.column_stack(smooth_data)
    flattened = raw_values.reshape(-1)
    num_feat = len(cols)
    real_window_len = window_size * num_feat
    
    if len(flattened) < real_window_len: return None

    features = sliding_window_view(flattened, window_shape=real_window_len)[::num_feat]
    return features

# ==========================================
# 2. 专家融合推断核心逻辑
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, default="../data/data4_hour_workday.csv")
    parser.add_argument("--anno_path", type=str, default="../data/anno_data9.0_2021.xlsx")
    parser.add_argument("--label_map_path", type=str, default="../data/label.json")
    parser.add_argument("--model_dir", type=str, default="./saved_models_optuna")
    parser.add_argument("--output_excel", type=str, default="expert_voting_report.xlsx")
    args = parser.parse_args()

    logger = setup_logger(args.model_dir)

    config_path = Path(args.model_dir) / "model_configs.json"
    if not config_path.exists():
        logger.error("Config not found. Please run train.py first.")
        return

    with open(config_path, "r") as f:
        model_configs = json.load(f)

    with open(args.label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    
    id_to_name = {v: k for k, v in label_map.items()}
    max_class_id = max(label_map.values())
    
    models = {}
    mappings = {} 
    model_names = ["p8p9f9", "p8"]
    
    for m_name in model_names:
        if m_name in model_configs and Path(model_configs[m_name]["model_path"]).exists():
            cfg = model_configs[m_name]
            models[m_name] = joblib.load(cfg["model_path"])
            mappings[m_name] = {int(k): int(v) for k, v in cfg["new_to_old"].items()}
            logger.info(f"Loaded Expert {m_name} (Responsible for: {list(mappings[m_name].values())})")
        else:
            logger.warning(f"Expert Model {m_name} not found.")

    if not models:
        logger.error("No models loaded.")
        return

    logger.info("Loading Test Data...")
    df_test, annodata = load_dataset(args.test_data, args.anno_path, label_map)
    days = sorted(list(set(df_test['day'])))

    full_preds, full_labels = [], []

    logger.info("Starting Expert Soft Voting Inference...")

    for t in tqdm(days, desc="Testing"):
        day_data = df_test[df_test['day'] == t].copy().reset_index(drop=True)
        N = len(day_data)

        # 构建 GT
        labels = np.zeros(N, dtype=int)
        for _, row in annodata[annodata['time'] == t].iterrows():
            if pd.isna(row.get('typea')): continue
            s, e, tp = int(row['start']), int(row['end']), int(row['typea'])
            s, e = max(0, s), min(N-1, e)
            if e >= s: labels[s:e+1] = tp

        # === 核心：构建全局概率矩阵 ===
        global_probs = np.zeros((N, max_class_id + 1), dtype=np.float32)

        for m_name, model in models.items():
            cfg = model_configs[m_name]
            X_feat = create_features_inference(day_data, cfg['cols'], cfg['sig'], cfg['win'])
            
            if X_feat is not None and len(X_feat) > 0:
                local_probs = model.predict_proba(X_feat)
                pad_len = N - len(local_probs)
                new_to_old_map = mappings[m_name]
                
                for new_idx, old_idx in new_to_old_map.items():
                    if old_idx > max_class_id: continue
                    prob_vec = local_probs[:, new_idx]
                    
                    if old_idx == 0:
                        # 【修改点】对于背景类(0)，取所有模型的平均概率
                        if pad_len > 0:
                            global_probs[pad_len:, 0] += prob_vec / len(models)
                        else:
                            global_probs[:N, 0] += prob_vec[:N] / len(models)
                    else:
                        # 【修改点】对于异常类，直接由专属“专家模型”的概率覆盖/累加
                        if pad_len > 0:
                            global_probs[pad_len:, old_idx] += prob_vec
                        else:
                            global_probs[:N, old_idx] += prob_vec

        # 决策：取全局概率和最大的类别
        final_pred = np.argmax(global_probs, axis=1)

        full_preds.extend(final_pred)
        full_labels.extend(labels)

    full_labels = np.array(full_labels)
    full_preds = np.array(full_preds)
    
    unique_ids = sorted(list(set(full_labels) | set(full_preds)))
    if 0 in unique_ids: unique_ids.remove(0)
    target_names = [id_to_name.get(i, f"Type {i}") for i in unique_ids]

    print("\n" + "#"*50)
    print("EXPERT VOTING TEST REPORT (DATA4)")
    print("#"*50)
    print(classification_report(full_labels, full_preds, labels=unique_ids, target_names=target_names, digits=4))

    # Save Excel
    report_dict = classification_report(full_labels, full_preds, labels=unique_ids, target_names=target_names, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose().round(4)
    save_path = Path(args.output_excel)
    df_report.to_excel(save_path)
    logger.info(f"Report saved to {save_path.absolute()}")

if __name__ == "__main__":
    main()
