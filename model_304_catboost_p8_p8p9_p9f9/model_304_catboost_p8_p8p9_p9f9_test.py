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
from sklearn.metrics import classification_report, confusion_matrix
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

def load_dataset(path, anno_path, label_map, other_id=None):
    df = pd.read_csv(path)
    if "8p" in df.columns:
        df = df.rename(columns={"time": "date", "8p": "p8", "9p": "p9", "9f": "f9"})
        
    df = df[["date", "p8", "p9", "f9"]]
    df = df.ffill()
    df['date'] = pd.to_datetime(df['date'].astype(str).apply(lambda x: x.split(".")[0]))
    df['day'] = df['date'].dt.date.apply(lambda x: x.replace(year=2025))
    
    annodata = pd.read_excel(anno_path)
    annodata['time'] = pd.to_datetime(annodata['time']).dt.date
    annodata['typea'] = annodata['type'].map(label_map)
    
    # 清洗 Other -> 0
    if other_id is not None:
        mask = annodata['typea'] == other_id
        if mask.any():
            annodata.loc[mask, 'typea'] = 0
            
    return df, annodata

def create_features_inference(df_segment, cols, sigma, window_size):
    smooth_data = []
    for c in cols:
        if sigma > 0: smooth_data.append(gaussian_filter(df_segment[c], sigma=sigma))
        else: smooth_data.append(df_segment[c].values)
    
    raw_values = np.column_stack(smooth_data)
    flattened = raw_values.reshape(-1)
    num_feat = len(cols)
    real_window_len = window_size * num_feat
    
    if len(flattened) < real_window_len: return None
    
    features = sliding_window_view(flattened, window_shape=real_window_len)[::num_feat]
    return features

def adjust_predicts(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    predicted = predicted.copy()
    anomaly_state = False
    for i in range(len(actual)):
        flag = predicted[i]
        if actual[i] == predicted[i] != 0 and not anomaly_state:
            anomaly_state = True
            for j in range(i, -1, -1):
                if actual[j] == 0 or actual[j] != flag: break
                if predicted[j] == 0: predicted[j] = flag
            for j in range(i, len(actual)):
                if actual[j] == 0 or actual[j] != flag: break
                if predicted[j] == 0: predicted[j] = flag
        elif actual[i] == 0:
            anomaly_state = False
        if anomaly_state: predicted[i] = flag
    return predicted

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, default="../data/data4.csv")
    parser.add_argument("--anno_path", type=str, default="../data/anno_data9.0.xlsx")
    parser.add_argument("--label_map_path", type=str, default="../data/label.json")
    parser.add_argument("--model_dir", type=str, default="./saved_models_catboost_clean")
    parser.add_argument("--output_excel", type=str, default="test_report_catboost_clean.xlsx")
    args, unknown = parser.parse_known_args()

    logger = setup_logger(args.model_dir)
    
    config_path = Path(args.model_dir) / "model_configs.json"
    if not config_path.exists():
        logger.error(f"Config not found at {config_path}. Run train code first.")
        return

    with open(config_path, "r") as f: model_configs = json.load(f)
    with open(args.label_map_path, 'r', encoding='utf-8') as f: label_map = json.load(f)
    
    # 识别 Other ID
    other_id = label_map.get("其他", None)
    id_to_name = {v: k for k, v in label_map.items() if v != other_id}
    
    models = {}
    for m_name, cfg in model_configs.items():
        if Path(cfg["model_path"]).exists():
            models[m_name] = joblib.load(cfg["model_path"])
            logger.info(f"Loaded {m_name}")

    logger.info("Loading Test Data (Cleaning Other)...")
    df_test, annodata = load_dataset(args.test_data, args.anno_path, label_map, other_id)
    days = sorted(list(set(df_test['day'])))
    
    full_preds, full_labels = [], []
    model_types = ["p8", "p8p9", "p9f9"] # 纯净列表

    logger.info("Starting Inference...")
    
    for t in tqdm(days, desc="Testing"):
        day_data = df_test[df_test['day'] == t].copy().reset_index(drop=True)
        N = len(day_data)
        
        labels = np.zeros(N, dtype=int)
        for _, row in annodata[annodata['time'] == t].iterrows():
            if pd.isna(row.get('typea')): continue
            s, e, tp = int(row['start']), int(row['end']), int(row['typea'])
            s, e = max(0, s), min(N-1, e)
            if e >= s: labels[s:e+1] = tp
        
        res_df = pd.DataFrame(0, index=range(N), columns=model_types)
        
        for m_name in model_types:
            if m_name not in models: continue
            cfg = model_configs[m_name]
            X_feat = create_features_inference(day_data, cfg['cols'], cfg['sig'], cfg['win'])
            
            if X_feat is not None:
                # CatBoost output shape handling
                raw_p = models[m_name].predict(X_feat)
                raw_p = np.array(raw_p).flatten().astype(int)
                
                mapped = np.zeros_like(raw_p)
                if m_name == "p8":
                    mapping = {0:0, 1:3, 2:4, 3:5}
                    mapped = np.vectorize(mapping.get)(raw_p)
                elif m_name == "p8p9":
                    mapped = np.where(raw_p==1, 1, 0)
                elif m_name == "p9f9":
                    mapping = {0:0, 1:2, 2:6}
                    mapped = np.vectorize(mapping.get)(raw_p)
                
                pad = N - len(mapped)
                if pad > 0:
                    res_df[m_name] = np.concatenate([np.zeros(pad, dtype=int), mapped])
                else:
                    res_df[m_name] = mapped

        # --- 融合逻辑 (Clean版) ---
        p8_v = res_df['p8'].values
        p8p9_v = res_df['p8p9'].values
        p9f9_v = res_df['p9f9'].values

        # 1. p8p9(1类) vs p9f9(2/6类) 冲突 -> 优先 p9f9 (假设p9f9更准)
        mask_310 = (p8_v != 0) & (p8p9_v == 1) & (p9f9_v == 0)
        p8p9_v[mask_310] = 0
        
        # 2. 复杂冲突
        mask_complex = (p8_v != 0) & (p8p9_v != 0) & (p9f9_v == 2)
        p8_v[mask_complex] = 0
        p8p9_v[mask_complex] = 0
        
        # 3. p8 缺失时的 p8p9
        mask_016 = (p8_v == 0) & (p8p9_v == 1) & (p9f9_v != 0)
        p8p9_v[mask_016] = 0
        
        final_res = np.maximum(np.maximum(p8_v, p8p9_v), p9f9_v)
        
        full_preds.extend(final_res)
        full_labels.extend(labels)

    full_labels = np.array(full_labels)
    full_preds = np.array(full_preds)
    
    preds_adj = adjust_predicts(full_labels, full_preds)
    
    unique_ids = sorted(list(set(full_labels) | set(preds_adj)))
    unique_ids = [i for i in unique_ids if i != 0 and i != other_id]
    target_names = [id_to_name.get(i, f"Type {i}") for i in unique_ids]
    
    print("\n" + "#"*40)
    print("FINAL TEST REPORT (CatBoost Clean)")
    print("#"*40)
    print(classification_report(full_labels, preds_adj, labels=unique_ids, target_names=target_names, digits=4))
    
    # Save Excel
    cm = confusion_matrix(full_labels, preds_adj, labels=unique_ids)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    
    report_dict = classification_report(full_labels, preds_adj, labels=unique_ids, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(4)
    
    save_path = Path(args.output_excel)
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        cm_df.to_excel(writer, sheet_name="Confusion Matrix")
        report_df.to_excel(writer, sheet_name="Report")
    
    logger.info(f"Excel saved to {save_path.absolute()}")

if __name__ == "__main__":
    main()
