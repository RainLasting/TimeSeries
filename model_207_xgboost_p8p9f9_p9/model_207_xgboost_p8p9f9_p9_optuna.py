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
from sklearn.metrics import f1_score
import xgboost as xgb
import optuna
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, cross_val_score, StratifiedKFold
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING) # 减少 Optuna 日志干扰

# ==========================================
# 1. 模型定义
# ==========================================

MODEL_DEFINITIONS = {
    "p8p9f9": {
        "cols": ["p8", "p9", "f9"],
        "objective": "multi:softprob", # 软投票必须用软概率输出
        "is_binary": False
    },
    "p9": {
        "cols": ["p9"],
        "objective": "multi:softprob",
        "is_binary": False
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
        fh = logging.FileHandler(Path(output_dir) / "training_optuna.log", mode='w', encoding='utf-8')
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
    df['day'] = df['date'].dt.date.apply(lambda x: x.replace(year=2025))

    annodata = pd.read_excel(anno_path)
    annodata['time'] = pd.to_datetime(annodata['time']).dt.date
    annodata['typea'] = annodata['type'].map(label_map)
    return df, annodata

def build_contiguous_class_mapping_global(label_map):
    """全局类别映射，保证软投票维度对齐"""
    defined_ids = set(label_map.values())
    if 0 not in defined_ids: defined_ids.add(0)
    all_labels = sorted(list(defined_ids))
    old_to_new = {old: new for new, old in enumerate(all_labels)}
    new_to_old = {new: old for old, new in old_to_new.items()}
    num_class = len(all_labels)
    return old_to_new, new_to_old, num_class

def get_labeled_raw_data_all_classes(df, annodata, feature_cols, old_to_new):
    """提取原始时间序列，不进行分窗"""
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
    """动态特征构建"""
    if raw_df.empty: return np.array([]), np.array([])
    
    # 1. 高斯平滑
    df_temp = pd.DataFrame()
    for col in feature_cols:
        if sigma > 0:
            df_temp[col] = gaussian_filter(raw_df[col], sigma=sigma)
        else:
            df_temp[col] = raw_df[col]

    if len(raw_df) < window_size: return np.array([]), np.array([])

    label = raw_df['anno'].values[window_size - 1:]
    raw_vals = df_temp[feature_cols].values
    num_feat = len(feature_cols)

    flattened = raw_vals.reshape(-1)
    real_window_len = window_size * num_feat
    slice_step = num_feat * step_size

    # 2. 滑动窗口 
    X = sliding_window_view(flattened, window_shape=real_window_len)[::slice_step]
    y = label[::step_size]

    min_len = min(len(X), len(y))
    return X[:min_len], y[:min_len]

# ==========================================
# 3. Optuna 目标函数
# ==========================================

def objective(trial, raw_df, feature_cols, num_class, device):
    # --- A. 搜索特征参数 ---
    win = trial.suggest_int('window_size', 50, 400, step=10)
    sig = trial.suggest_int('sigma', 5, 100, step=5)
    # 步长率：步长 = 窗口大小 * step_rate (例如 10%~30%)
    step_rate = trial.suggest_float('step_rate', 0.05, 0.3)
    actual_step = max(1, int(win * step_rate))

    # 构建数据 (最耗时步骤)
    X, y = build_features(raw_df, feature_cols, sig, win, actual_step)
    
    if len(y) < 100: # 样本太少直接剪枝
        return 0.0

    # --- B. 搜索 XGBoost 参数 ---
    params = {
        'objective': 'multi:softprob',
        'num_class': num_class,
        'tree_method': 'hist',
        'device': device,
        'booster': 'gbtree',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'verbosity': 0,
        'n_jobs': -1
    }

    # GPU 兼容性处理
    if device == 'cuda':
        try:
            if float(xgb.__version__.split('.')[0]) < 2:
                params['tree_method'] = 'gpu_hist'
                del params['device']
        except: pass

    # --- C. 交叉验证 ---
    # 使用 f1_macro 以应对类别不平衡
    clf = xgb.XGBClassifier(**params)
    
    # 3折交叉验证，速度优先
    #cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv = TimeSeriesSplit(n_splits=5)
    
    try:
        scores = cross_val_score(clf, X, y, cv=cv, scoring='f1_macro')
        return scores.mean()
    except Exception:
        return 0.0

# ==========================================
# 4. 主程序
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="../data/data3_hour_workday.csv")
    parser.add_argument("--anno_path", type=str, default="../data/anno_data9.0_2021.xlsx")
    parser.add_argument("--label_map_path", type=str, default="../data/label.json")
    parser.add_argument("--output_dir", type=str, default="./saved_models_optuna")
    parser.add_argument("--n_trials", type=int, default=2, help="Number of trials per model")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()

    logger = setup_logger(args.output_dir)

    # 1. 加载映射与数据
    with open(args.label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    
    df_train, annodata = load_dataset(args.train_data, args.anno_path, label_map)
    
    # 全局映射构建 (确保软投票对齐)
    old_to_new, new_to_old, num_class = build_contiguous_class_mapping_global(label_map)
    logger.info(f"Global Class Mapping: {num_class} classes. Mapping: {new_to_old}")

    final_configs = {}
    total_start = time.time()

    # 2. 逐个模型优化
    for model_name, def_cfg in MODEL_DEFINITIONS.items():
        logger.info(f"{'='*30}")
        logger.info(f"Optimizing Model: {model_name}")
        
        # 准备原始数据 (不分窗)
        raw_df = get_labeled_raw_data_all_classes(df_train, annodata, def_cfg["cols"], old_to_new)
        
        if raw_df.empty:
            logger.warning(f"No data for {model_name}, skipping.")
            continue
            
        logger.info(f"Raw data prepared. Rows: {len(raw_df)}. Starting Optuna...")

        # 创建 Study
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: objective(trial, raw_df, def_cfg["cols"], num_class, args.device),
            n_trials=args.n_trials
        )

        best_params = study.best_params
        best_score = study.best_value
        logger.info(f"Best F1-Macro: {best_score:.4f}")
        logger.info(f"Best Params: {best_params}")

        # 3. 使用最佳参数重新训练
        logger.info("Retraining final model with best parameters...")
        
        # 提取特征参数
        win = best_params.pop('window_size')
        sig = best_params.pop('sigma')
        step_rate = best_params.pop('step_rate')
        actual_step = max(1, int(win * step_rate))

        # 剩下的就是 XGBoost 参数
        xgb_params = best_params.copy()
        xgb_params.update({
            'objective': 'multi:softprob',
            'num_class': num_class,
            'tree_method': 'hist',
            'device': args.device,
            'n_jobs': -1
        })
        
        if args.device == 'cuda':
            try:
                if float(xgb.__version__.split('.')[0]) < 2:
                    xgb_params['tree_method'] = 'gpu_hist'
                    del xgb_params['device']
            except: pass

        # 重新构建特征 (全量数据)
        X_final, y_final = build_features(raw_df, def_cfg["cols"], sig, win, actual_step)
        
        # 训练
        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X_final, y_final)

        # 保存
        save_path = Path(args.output_dir) / f"{model_name}.joblib"
        joblib.dump(clf, save_path)
        
        final_configs[model_name] = {
            "model_path": str(save_path),
            "cols": def_cfg["cols"],
            "win": win,
            "sig": sig,
            "actual_step": actual_step, # 仅供参考，测试时通常 step=1
            "num_class": num_class,
            "new_to_old": new_to_old,
            "cv_score": best_score,
            "best_params": best_params
        }
        logger.info(f"Model saved to {save_path}")

    # 保存总配置
    config_path = Path(args.output_dir) / "model_configs.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(final_configs, f, indent=4)

    logger.info(f"All tasks finished in {time.time() - total_start:.2f}s")

if __name__ == "__main__":
    main()
