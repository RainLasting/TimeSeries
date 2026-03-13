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
# 1. 模型定义与任务分工
# ==========================================
# "反冲洗": 1, "9井关井": 2, "短暂放气": 3, "设备启动": 4, "设备停机": 5, "9井开井": 6

MODEL_DEFINITIONS = {
    # 专家 1：看 3 个变量，只负责判断 1(反冲洗), 2(关井), 6(开井)
    "p8p9f9": {
        "cols": ["p8", "p9", "f9"],
        "target_classes": [1, 2, 6],
        "objective": "multi:softprob", 
        "is_binary": False
    },
    # 专家 2：只看 P8，只负责判断 3(短暂放气), 4(设备启动), 5(设备停机)
    "p8": {
        "cols": ["p8"],
        "target_classes": [3, 4, 5],
        "objective": "multi:softprob",
        "is_binary": False
    }
}

# 固定的超参数配置 (直接使用你之前 Optuna 或手动调优出的最佳参数)
HYPERPARAMS = {
    "p8p9f9": {
            "window_size": 150,
            "sigma": 90,
            "actual_step": 8,
            "n_estimators": 700,
            "max_depth": 6,
            "learning_rate": 0.16026002748977755,
            "subsample": 0.6772432383821406,
            "colsample_bytree": 0.9139403229554397,
            "min_child_weight": 1,
            "gamma": 1.7488251268811075,
            "reg_alpha": 0.01640328102507082,
            "reg_lambda": 0.0002497828561420395
    },
    "p8": {
            "window_size": 310,
            "sigma": 45,
            "actual_step": 16,
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1301508270243619,
            "subsample": 0.8571163537553046,
            "colsample_bytree": 0.8408534567771128,
            "min_child_weight": 6,
            "gamma": 3.814340318673825,
            "reg_alpha": 1.918170837105672e-06,
            "reg_lambda": 0.37058391909607385
    }
}

# ==========================================
# 2. 数据处理与特征构建
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
    #.apply(lambda x: x.replace(year=2025)) # 如有需要可取消注释

    annodata = pd.read_excel(anno_path)
    annodata['time'] = pd.to_datetime(annodata['time']).dt.date
    annodata['typea'] = annodata['type'].map(label_map)
    return df, annodata

def build_local_class_mapping(target_classes):
    """为当前模型建立专属的连续类别映射"""
    all_labels = [0] + sorted(target_classes)
    old_to_new = {old: new for new, old in enumerate(all_labels)}
    new_to_old = {new: old for old, new in old_to_new.items()}
    num_class = len(all_labels)
    return old_to_new, new_to_old, num_class

def get_labeled_raw_data_target_classes(df, annodata, feature_cols, old_to_new):
    """过滤标注：只提取当前模型负责的类别，其他异常全部视为背景 0"""
    days = sorted(list(set(df["day"])))
    data_list = []
    for t in days:
        sub = df[df['day'] == t][feature_cols].copy()
        sub['anno'] = 0
        sub.reset_index(drop=True, inplace=True)

        day_annos = annodata[annodata['time'] == t]
        for _, row in day_annos.iterrows():
            if pd.isna(row.get('typea')): continue
            s, e = int(row['start']), int(row['end'])
            s, e = max(0, s), min(len(sub) - 1, e)
            old_label = int(row['typea'])
            
            # 如果标签在当前模型的负责范围内 (在 old_to_new 字典中)
            if old_label in old_to_new:
                new_label = old_to_new[old_label]
                if e >= s:
                    sub.loc[s:e, 'anno'] = new_label
                    
        data_list.append(sub)

    if not data_list: return pd.DataFrame()
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
# 3. 训练主程序
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="../data/data3_hour_workday.csv")
    parser.add_argument("--anno_path", type=str, default="../data/anno_data9.0_2021.xlsx")
    parser.add_argument("--label_map_path", type=str, default="../data/label_new.json")
    parser.add_argument("--output_dir", type=str, default="./saved_models")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()

    logger = setup_logger(args.output_dir)

    with open(args.label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    logger.info("Label Map Loaded.")

    logger.info("Loading Training Data...")
    df_train, annodata = load_dataset(args.train_data, args.anno_path, label_map)

    final_configs = {}
    start_all_train = time.time()

    # 循环训练每一个专家模型
    for model_name, def_cfg in MODEL_DEFINITIONS.items():
        logger.info(f"\n{'='*40}")
        logger.info(f"--- Training Expert Model: {model_name} ---")
        logger.info(f"Target Classes: {def_cfg['target_classes']}")
        
        # 提取当前模型的超参数
        params = HYPERPARAMS[model_name].copy()
        win = params.pop("window_size")
        sig = params.pop("sigma")
        actual_step = params.pop("actual_step")

        # 为当前模型构建专属局部映射
        old_to_new, new_to_old, num_class = build_local_class_mapping(def_cfg["target_classes"])
        logger.info(f"Local Class Mapping (New -> Old): {new_to_old} (Total {num_class} classes)")

        # 过滤数据：非目标类的异常全视为正常(0)
        raw_df = get_labeled_raw_data_target_classes(
            df_train, annodata, def_cfg["cols"], old_to_new
        )

        if raw_df.empty:
            logger.error(f"No training data produced for {model_name}; abort.")
            continue

        X, y = build_features(raw_df, def_cfg["cols"], sig, win, actual_step)
        if len(X) == 0:
            logger.error(f"Not enough samples for windowing for {model_name}; abort.")
            continue

        # 检查训练集中是否包含所有负责的标签
        unique_y = np.unique(y)
        missing_labels = set(range(num_class)) - set(unique_y)
        if missing_labels:
            logger.warning(f"Warning: Model {model_name} training data is missing mapped classes: {missing_labels}.")

        xgb_params = params.copy()
        xgb_params.update({
            "objective": def_cfg["objective"], 
            "num_class": num_class, 
            "tree_method": "hist",
            "device": args.device,
            "n_jobs": -1
        })
        
        # GPU 兼容性处理
        if args.device == 'cuda':
            try:
                if float(xgb.__version__.split('.')[0]) < 2:
                    xgb_params['tree_method'] = 'gpu_hist'
                    del xgb_params['device']
            except: pass

        start_single_model = time.time()
        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X, y)
        duration = time.time() - start_single_model

        save_path = Path(args.output_dir) / f"{model_name}.joblib"
        joblib.dump(clf, save_path)
        logger.info(f"Model {model_name} saved to {save_path} (train_time={duration:.2f}s)")

        # 保存配置 (测试脚本会依赖 new_to_old 来进行软投票矩阵拼图)
        final_configs[model_name] = {
            "model_path": str(save_path),
            "cols": def_cfg["cols"],
            "win": win,
            "sig": sig,
            "actual_step": actual_step,
            "num_class": num_class,
            "new_to_old": new_to_old
        }

    config_path = Path(args.output_dir) / "model_configs.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(final_configs, f, indent=4)

    total_duration = time.time() - start_all_train
    logger.info(f"\nTotal training process took: {total_duration:.2f} seconds")
    logger.info("Training Complete.")

if __name__ == "__main__":
    main()
