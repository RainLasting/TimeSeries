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
import optuna
import xgboost as xgb

from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import f1_score
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================
# 1. 专家模型定义与任务分工
# ==========================================
MODEL_DEFINITIONS = {
    "p8p9f9": {
        "cols": ["p8", "p9", "f9", "hour", "is_workday"],
        "target_classes": [1, 2, 6],
        "objective": "multi:softprob",
        "is_binary": False
    },
    "p8": {
        "cols": ["p8", "hour", "is_workday"],
        "target_classes": [3, 4, 5],
        "objective": "multi:softprob",
        "is_binary": False
    }
}

def setup_logger(output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("TrainOptunaLogger")
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
    df['day'] = df['date'].dt.date
    annodata = pd.read_excel(anno_path)
    annodata['time'] = pd.to_datetime(annodata['time']).dt.date
    annodata['typea'] = annodata['type'].map(label_map)
    return df, annodata

def build_local_class_mapping(target_classes):
    all_labels = [0] + sorted(target_classes)
    old_to_new = {old: new for new, old in enumerate(all_labels)}
    new_to_old = {new: old for old, new in old_to_new.items()}
    num_class = len(all_labels)
    return old_to_new, new_to_old, num_class

# ==========================================
# 2. 关键修改：提取数据时生成 groups (按天分组)
# ==========================================
def get_labeled_raw_data_target_classes(df, annodata, feature_cols, old_to_new):
    days = sorted(list(set(df["day"])))
    data_list = []
    group_list = []

    for gi, t in enumerate(days):
        sub = df[df["day"] == t][feature_cols].copy()
        sub['anno'] = 0
        sub.reset_index(drop=True, inplace=True)

        day_annos = annodata[annodata["time"] == t]
        for _, row in day_annos.iterrows():
            if pd.isna(row.get('typea')): continue
            s, e = int(row['start']), int(row['end'])
            s, e = max(0, s), min(len(sub) - 1, e)
            old_label = int(row['typea'])
            if old_label in old_to_new:
                sub.loc[s:e, "anno"] = old_to_new[old_label]
                    
        data_list.append(sub)
        group_list.append(np.full(len(sub), gi, dtype=int)) # 生成对应的组 ID

    if not data_list: return pd.DataFrame(), None
    full_df = pd.concat(data_list, axis=0).reset_index(drop=True)
    full_df['anno'] = full_df['anno'].astype(int)
    groups = np.concatenate(group_list, axis=0)
    return full_df, groups

# ==========================================
# 3. 关键修改：特征构建函数对齐 groups
# ==========================================
def build_features_with_groups(raw_df, groups, feature_cols, sigma, window_size, step_size):
    if raw_df.empty or groups is None:
        return np.array([]), np.array([]), np.array([])

    df_temp = pd.DataFrame()
    for col in feature_cols:
        if sigma > 0:
            df_temp[col] = gaussian_filter(raw_df[col].values, sigma=sigma)
        else:
            df_temp[col] = raw_df[col].values

    if len(raw_df) < window_size:
        return np.array([]), np.array([]), np.array([])

    y_all = raw_df["anno"].values[window_size - 1:]
    g_all = groups[window_size - 1:]

    raw_vals = df_temp[feature_cols].values
    num_feat = len(feature_cols)
    flattened = raw_vals.reshape(-1)
    real_window_len = window_size * num_feat
    slice_step = num_feat * step_size

    X = sliding_window_view(flattened, window_shape=real_window_len)[::slice_step]
    y = y_all[::step_size]
    g = g_all[::step_size]

    m = min(len(X), len(y), len(g))
    return X[:m], y[:m], g[:m]

# ==========================================
# 4. Optuna 目标函数
# ==========================================
def objective(trial, raw_df, groups_raw, feature_cols, num_class, device):
    win = trial.suggest_int('window_size', 50, 400, step=10)
    sig = trial.suggest_int('sigma', 5, 100, step=5)
    step_rate = trial.suggest_float('step_rate', 0.05, 0.3)
    actual_step = max(1, int(win * step_rate))

    # 正确调用，获取 g 变量
    X, y, g = build_features_with_groups(raw_df, groups_raw, feature_cols, sig, win, actual_step)
    
    if len(y) < 100: return 0.0

    params = {
        'objective': 'multi:softprob',
        'num_class': num_class,
        'tree_method': 'hist',
        'device': device,
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True),
        'verbosity': 0,
        'n_jobs': -1
    }

    if device == 'cuda':
        try:
            if float(xgb.__version__.split('.')[0]) < 2:
                params['tree_method'] = 'gpu_hist'
                del params['device']
        except: pass

    gkf = GroupKFold(n_splits=5)
    f1s = []
    try:
        # 这里的 g 现在已经通过 build_features_with_groups 获取了
        for tr_idx, va_idx in gkf.split(X, y, groups=g):
            clf = xgb.XGBClassifier(**params)
            clf.fit(X[tr_idx], y[tr_idx])
            pred = clf.predict(X[va_idx])
            f1s.append(f1_score(y[va_idx], pred, average="macro"))
        return float(np.mean(f1s))
    except Exception:
        return 0.0

# ==========================================
# 5. 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="../data/data3_hour_workday.csv")
    parser.add_argument("--anno_path", type=str, default="../data/anno_data9.0_2021.xlsx")
    parser.add_argument("--label_map_path", type=str, default="../data/label.json")
    parser.add_argument("--output_dir", type=str, default="./saved_models_optuna")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()

    logger = setup_logger(args.output_dir)
    with open(args.label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    
    df_train, annodata = load_dataset(args.train_data, args.anno_path, label_map)
    
    final_configs = {}
    total_start = time.time()

    for model_name, def_cfg in MODEL_DEFINITIONS.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"专家模型优化: {model_name}")
        
        old_to_new, new_to_old, num_class = build_local_class_mapping(def_cfg["target_classes"])

        # 接收两个返回值：raw_df 和 groups_raw
        raw_df, groups_raw = get_labeled_raw_data_target_classes(df_train, annodata, def_cfg["cols"], old_to_new)
        
        if raw_df.empty:
            continue
            
        study = optuna.create_study(direction='maximize')
        study.optimize(
            # 传入 groups_raw 给目标函数
            lambda trial: objective(trial, raw_df, groups_raw, def_cfg["cols"], num_class, args.device),
            n_trials=args.n_trials
        )

        best_params = study.best_params
        best_score = study.best_value
        logger.info(f"Best Score: {best_score:.4f}")

        # 重训最终模型
        win, sig, step_rate = best_params.pop('window_size'), best_params.pop('sigma'), best_params.pop('step_rate')
        actual_step = max(1, int(win * step_rate))

        # 构建最终特征 (不带 groups，因为重训不需要分折)
        X_final, y_final, _ = build_features_with_groups(raw_df, groups_raw, def_cfg["cols"], sig, win, actual_step)
        
        xgb_params = best_params.copy()
        xgb_params.update({'objective': 'multi:softprob', 'num_class': num_class, 'tree_method': 'hist', 'device': args.device, 'n_jobs': -1})
        
        if args.device == 'cuda' and float(xgb.__version__.split('.')[0]) < 2:
            xgb_params['tree_method'] = 'gpu_hist'
            del xgb_params['device']

        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X_final, y_final)

        save_path = Path(args.output_dir) / f"{model_name}.joblib"
        joblib.dump(clf, save_path)
        
        final_configs[model_name] = {
            "model_path": str(save_path), "cols": def_cfg["cols"], "win": win, "sig": sig, 
            "actual_step": actual_step, "num_class": num_class, "new_to_old": new_to_old
        }

    with open(Path(args.output_dir) / "model_configs.json", "w", encoding="utf-8") as f:
        json.dump(final_configs, f, indent=4)

    logger.info(f"Finished in {time.time() - total_start:.2f}s")

if __name__ == "__main__":
    main()

