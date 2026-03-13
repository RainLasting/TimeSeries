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
import xgboost as xgb
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")

# ==========================================
# 1. all_with_hour 预测所有类
# ==========================================

MODEL_DEFINITIONS = {
    "all_with_hour": {
        "cols": ["p8", "p9", "f9", "hour"],
        # 预测所有类 -> 多分类概率输出
        "objective": "multi:softprob",
        "num_class": None,     # 运行时根据 label_map / 数据自动推断
        "is_binary": False
    }
}

HYPERPARAMS = {
    "all_with_hour": {
        "window_size": int(100 * 1.0), "sigma": int(13 * 1.0), "actual_step": int(10 * 1.0),
        "subsample": 0.8854, "colsample_bytree": 0.9898, "gamma": 3.3655,
        "reg_alpha": 0.9308, "reg_lambda": 0.5164, "n_estimators": 1200,
        "max_depth": 6, "learning_rate": 0.2798, "min_child_weight": 5,
        "max_delta_step": 7
    }
}

# ==========================================
# 2. 数据处理
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

    # annodata['type'] -> label_id（来自 label_map）
    annodata['typea'] = annodata['type'].map(label_map)

    return df, annodata

def build_contiguous_class_mapping(annodata):
    """
    把原始 label_id（可能是 1,2,6,10...）重映射为 contiguous: 0..K-1
    同时保留 0 作为背景类（无标注）——可选：你也可以选择不把 0 作为类，但那需要额外采样策略。
    """
    # 收集出现过的 label（去掉 NaN）
    labels = sorted({int(x) for x in annodata['typea'].dropna().unique()})

    # 0 作为背景类（无标注）
    all_labels = [0] + [lb for lb in labels if lb != 0]

    old_to_new = {old: new for new, old in enumerate(all_labels)}
    new_to_old = {new: old for old, new in old_to_new.items()}
    num_class = len(all_labels)
    return old_to_new, new_to_old, num_class

def get_labeled_raw_data_all_classes(df, annodata, feature_cols, old_to_new):
    """
    生成训练序列：anno 直接写成“所有类”的 id（contiguous 后的 0..K-1）
    """
    days = sorted(list(set(df["day"])))
    data_list = []

    for t in days:
        sub = df[df['day'] == t][feature_cols].copy()
        sub['anno'] = 0  # 默认背景类 0
        sub.reset_index(drop=True, inplace=True)

        day_annos = annodata[annodata['time'] == t]
        for _, row in day_annos.iterrows():
            if pd.isna(row.get('typea')):
                continue

            s, e = int(row['start']), int(row['end'])
            s, e = max(0, s), min(len(sub) - 1, e)

            old_label = int(row['typea'])
            # 如果 label 不在 mapping 中就跳过
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

def build_features(raw_df, feature_cols, sigma, window_size, step_size):
    df_temp = pd.DataFrame()
    for col in feature_cols:
        if sigma > 0:
            df_temp[col] = gaussian_filter(raw_df[col], sigma=sigma)
        else:
            df_temp[col] = raw_df[col]

    if len(raw_df) < window_size:
        return np.array([]), np.array([])

    # 用 window 末端的 label
    label = raw_df['anno'].values[window_size - 1:]
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
    parser.add_argument("--train_data", type=str, default="../data/data3_hour_workday.csv")
    parser.add_argument("--anno_path", type=str, default="../data/anno_data9.0_2021.xlsx")
    parser.add_argument("--label_map_path", type=str, default="../data/label.json")
    parser.add_argument("--output_dir", type=str, default="./saved_models")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()

    logger = setup_logger(args.output_dir)

    with open(args.label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    logger.info("Label Map Loaded.")

    logger.info("Loading Training Data...")
    df_train, annodata = load_dataset(args.train_data, args.anno_path, label_map)

    # 生成 contiguous label 映射，保证 XGBoost 多分类 label 从 0..K-1
    old_to_new, new_to_old, num_class = build_contiguous_class_mapping(annodata)
    logger.info(f"Detected num_class = {num_class} (including background=0).")

    final_configs = {}

    start_all_train = time.time()

    # all_with_hour
    model_name = "all_with_hour"
    def_cfg = MODEL_DEFINITIONS[model_name]
    logger.info(f"--- Training Model: {model_name} (ALL-CLASS) ---")

    params = HYPERPARAMS[model_name].copy()
    win = params.pop("window_size")
    sig = params.pop("sigma")
    actual_step = params.pop("actual_step")

    raw_df = get_labeled_raw_data_all_classes(
        df_train, annodata, def_cfg["cols"], old_to_new
    )

    if raw_df.empty:
        logger.error("No training data produced; abort.")
        return

    X, y = build_features(raw_df, def_cfg["cols"], sig, win, actual_step)
    if len(X) == 0:
        logger.error("Not enough samples for windowing; abort.")
        return

    xgb_params = params.copy()
    xgb_params.update({
        "objective": def_cfg["objective"],
        "num_class": num_class,
        "tree_method": "hist",
        "device": args.device
    })

    # 兼容旧版 xgboost GPU 参数
    if args.device == 'cuda':
        try:
            if float(xgb.__version__.split('.')[0]) < 2:
                xgb_params['tree_method'] = 'gpu_hist'
                del xgb_params['device']
        except:
            pass

    start_single_model = time.time()
    clf = xgb.XGBClassifier(**xgb_params)
    clf.fit(X, y)
    duration = time.time() - start_single_model

    save_path = Path(args.output_dir) / f"{model_name}.joblib"
    joblib.dump(clf, save_path)
    logger.info(f"Model saved to {save_path} (train_time={duration:.2f}s)")

    # 保存配置：包含 label 映射，预测时把输出类索引还原成原始 label_id
    config_path = Path(args.output_dir) / "model_configs.json"
    final_configs[model_name] = {
        "model_path": str(save_path),
        "cols": def_cfg["cols"],
        "win": win,
        "sig": sig,
        #"actual_step": actual_step,
        "num_class": num_class,
        "old_to_new": old_to_new,
        "new_to_old": new_to_old
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(final_configs, f, indent=4, ensure_ascii=False)

    total_duration = time.time() - start_all_train
    logger.info(f"Total training process took: {total_duration:.2f} seconds")
    logger.info("Training Complete.")

if __name__ == "__main__":
    main()

