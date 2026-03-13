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
    logger = logging.getLogger("SoftVotingTest")
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
    #.apply(lambda x: x.replace(year=2025))
    
    annodata = pd.read_excel(anno_path)
    annodata['time'] = pd.to_datetime(annodata['time']).dt.date
    annodata['typea'] = annodata['type'].map(label_map)
    return df, annodata

def create_features_inference(df_segment, cols, sigma, window_size):
    """
    推理特征构建，步长为1，保证输出密度
    """
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
    
    if len(flattened) < real_window_len:
        return None

    features = sliding_window_view(flattened, window_shape=real_window_len)[::num_feat]
    return features

def adjust_predicts(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """后处理：填补预测间隙，平滑结果"""
    predicted = predicted.copy()
    anomaly_state = False
    for i in range(len(actual)):
        flag = predicted[i]
        # 当遇到非0预测，且该预测与GT一致（或者单纯信任预测），进入异常状态
        if actual[i] == predicted[i] != 0 and not anomaly_state:
            anomaly_state = True
            # 回溯填补
            for j in range(i, -1, -1):
                if actual[j] == 0 or actual[j] != flag: break
                if predicted[j] == 0: predicted[j] = flag
            # 前瞻填补
            for j in range(i, len(actual)):
                if actual[j] == 0 or actual[j] != flag: break
                if predicted[j] == 0: predicted[j] = flag
        elif actual[i] == 0:
            anomaly_state = False
        if anomaly_state: predicted[i] = flag
    return predicted

# ==========================================
# 2. 软投票核心逻辑
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, default="../data/data4_hour_workday.csv")
    parser.add_argument("--anno_path", type=str, default="../data/anno_data9.0_2021.xlsx")
    parser.add_argument("--label_map_path", type=str, default="../data/label.json")
    parser.add_argument("--model_dir", type=str, default="./saved_models_optuna")
    parser.add_argument("--output_excel", type=str, default="soft_voting_report.xlsx")
    args = parser.parse_args()

    logger = setup_logger(args.model_dir)

    # 1. 加载配置
    config_path = Path(args.model_dir) / "model_configs.json"
    if not config_path.exists():
        logger.error("Config not found. Please run train.py first.")
        return

    with open(config_path, "r") as f:
        model_configs = json.load(f)

    with open(args.label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    
    # 类别ID转中文名称
    id_to_name = {v: k for k, v in label_map.items()}
    
    # 确定全局最大类别ID (通常是 7)
    max_class_id = max(label_map.values())
    logger.info(f"Global Max Class ID: {max_class_id}")

    # 2. 加载模型与映射表
    models = {}
    mappings = {} # 存储 new_to_old
    
    # 定义模型名称列表
    model_names = ["p8p9f9", "p9"]
    
    for m_name in model_names:
        if m_name in model_configs and Path(model_configs[m_name]["model_path"]).exists():
            cfg = model_configs[m_name]
            models[m_name] = joblib.load(cfg["model_path"])
            # JSON key 是 str, 需转 int
            mappings[m_name] = {int(k): int(v) for k, v in cfg["new_to_old"].items()}
            logger.info(f"Loaded {m_name} (Trained on classes: {list(mappings[m_name].values())})")
        else:
            logger.warning(f"Model {m_name} not found or missing config.")

    if not models:
        logger.error("No models loaded.")
        return

    # 3. 加载数据
    logger.info("Loading Test Data...")
    df_test, annodata = load_dataset(args.test_data, args.anno_path, label_map)
    days = sorted(list(set(df_test['day'])))

    full_preds = []
    full_labels = []

    logger.info("Starting Soft Voting Inference...")

    for t in tqdm(days, desc="Testing"):
        day_data = df_test[df_test['day'] == t].copy().reset_index(drop=True)
        N = len(day_data)

        # 构建 GT
        labels = np.zeros(N, dtype=int)
        for _, row in annodata[annodata['time'] == t].iterrows():
            s, e, tp = int(row['start']), int(row['end']), int(row['typea'])
            s, e = max(0, s), min(N-1, e)
            if e >= s: labels[s:e+1] = tp

        # === 核心：构建全局概率矩阵 ===
        # shape: (N, 8)，对应类别 0-7
        global_probs = np.zeros((N, max_class_id + 1), dtype=np.float32)

        for m_name, model in models.items():
            cfg = model_configs[m_name]
            X_feat = create_features_inference(day_data, cfg['cols'], cfg['sig'], cfg['win'])
            
            if X_feat is not None and len(X_feat) > 0:
                # 
                # 获取概率: shape (n_samples, n_local_classes)
                # predict_proba 返回的是模型内部类别的概率
                local_probs = model.predict_proba(X_feat)
                
                # 长度对齐：因为 sliding window，头部会有 NaN/空缺
                pad_len = N - len(local_probs)
                
                # 遍历模型的局部类别，映射到全局概率矩阵
                new_to_old_map = mappings[m_name]
                
                # new_idx 是模型输出的列索引，old_idx 是真实类别ID (0-7)
                for new_idx, old_idx in new_to_old_map.items():
                    if old_idx > max_class_id: continue # 防止越界
                    
                    # 取出该类别的概率向量
                    prob_vec = local_probs[:, new_idx]
                    
                    # 累加到全局矩阵对应的位置 (注意处理头部 padding)
                    if pad_len > 0:
                        # 头部缺失部分默认概率为0 (或者可以设为背景类概率，这里简化处理)
                        global_probs[pad_len:, old_idx] += prob_vec
                    else:
                        global_probs[:N, old_idx] += prob_vec[:N]
            else:
                # 特征构建失败（通常因为数据太短），不加概率
                pass

        # === 决策 ===
        # 1. 简单 Argmax: 取概率和最大的类别
        final_pred = np.argmax(global_probs, axis=1)
        
        # (可选) 阈值过滤：如果最大概率也很低，强制归为 0 (正常)
        # max_probs = np.max(global_probs, axis=1)
        # final_pred[max_probs < 1.5] = 0 # 假设3个模型，满分3.0，低于1.5说明都很犹豫

        full_preds.extend(final_pred)
        full_labels.extend(labels)

    # 4. 后处理与报告
    full_labels = np.array(full_labels)
    full_preds = np.array(full_preds)

    #preds_adj = adjust_predicts(full_labels, full_preds)
    preds_adj = full_preds
    unique_ids = sorted(list(set(full_labels) | set(preds_adj)))
    if 0 in unique_ids: unique_ids.remove(0)
    target_names = [id_to_name.get(i, f"Type {i}") for i in unique_ids]

    print("\n" + "#"*50)
    print("SOFT VOTING TEST REPORT (DATA4)")
    print("#"*50)
    print(classification_report(full_labels, preds_adj, labels=unique_ids, target_names=target_names, digits=4))

    # Save Excel
    report_dict = classification_report(full_labels, preds_adj, labels=unique_ids, target_names=target_names, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose().round(4)
    
    save_path = Path(args.output_excel)
    df_report.to_excel(save_path)
    logger.info(f"Report saved to {save_path.absolute()}")

if __name__ == "__main__":
    main()
