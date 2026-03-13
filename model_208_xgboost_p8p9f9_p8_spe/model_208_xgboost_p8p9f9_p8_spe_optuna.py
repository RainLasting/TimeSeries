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
optuna.logging.set_verbosity(optuna.logging.WARNING) # 减少 Optuna 的刷屏日志干扰

# ==========================================
# 1. 专家模型定义与任务分工
# ==========================================
# 原始 Label 对应关系:
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

# ==========================================
# 2. 数据处理与特征构建
# ==========================================

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
    """动态特征构建"""
    if raw_df.empty: return np.array([]), np.array([])
    
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

    X = sliding_window_view(flattened, window_shape=real_window_len)[::slice_step]
    y = label[::step_size]

    min_len = min(len(X), len(y))
    return X[:min_len], y[:min_len]

# ==========================================
# 3. Optuna 目标函数 (自动调参逻辑)
# ==========================================

def objective(trial, raw_df, feature_cols, num_class, device):
    # A. 搜索特征参数 (窗口大小、平滑度、滑动步长比例)
    win = trial.suggest_int('window_size', 50, 400, step=10)
    sig = trial.suggest_int('sigma', 5, 100, step=5)
    step_rate = trial.suggest_float('step_rate', 0.05, 0.3)
    actual_step = max(1, int(win * step_rate))

    X, y = build_features(raw_df, feature_cols, sig, win, actual_step)
    
    if len(y) < 100: return 0.0 # 样本太少直接放弃这一组参数

    # B. 搜索 XGBoost 树模型参数
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

    # ---- C) GroupKFold 按天交叉验证，macro-F1 ----
    gkf = GroupKFold(n_splits=5)
    f1s = []
    try:
        for tr_idx, va_idx in gkf.split(X, y, groups=g):
            clf = xgb.XGBClassifier(**params)
            clf.fit(X[tr_idx], y[tr_idx])

            pred = clf.predict(X[va_idx])
            f1s.append(f1_score(y[va_idx], pred, average="macro"))
        return float(np.mean(f1s))
    except Exception:
        # 比如 GPU OOM 等
        return 0.0

# ==========================================
# 4. 主程序：自动调参 -> 挑选最佳参数 -> 训练最终模型 -> 保存
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="../data/data3_hour_workday.csv")
    parser.add_argument("--anno_path", type=str, default="../data/anno_data9.0_2021.xlsx")
    parser.add_argument("--label_map_path", type=str, default="../data/label.json")
    parser.add_argument("--output_dir", type=str, default="./saved_models_optuna")
    parser.add_argument("--n_trials", type=int, default=2, help="每个模型让 Optuna 搜索多少次组合")
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
        logger.info(f"开始优化并训练专家模型: {model_name}")
        logger.info(f"该模型负责的原始类别 (Target Classes): {def_cfg['target_classes']}")
        
        # 1. 建立局部映射
        old_to_new, new_to_old, num_class = build_local_class_mapping(def_cfg["target_classes"])
        logger.info(f"局部类别映射 (New -> Old): {new_to_old} (共 {num_class} 个分类项，含背景 0)")

        # 2. 提取并过滤该模型关注的数据
        raw_df = get_labeled_raw_data_target_classes(df_train, annodata, def_cfg["cols"], old_to_new)
        
        if raw_df.empty:
            logger.warning(f"没有找到 {model_name} 需要的训练数据，跳过。")
            continue
            
        logger.info(f"原始数据提取完毕，共 {len(raw_df)} 行。启动 Optuna 超参数搜索 (共 {args.n_trials} 轮)...")

        # 3. 让 Optuna 开始寻找最佳特征窗口和 XGB 参数
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: objective(trial, raw_df, def_cfg["cols"], num_class, args.device),
            n_trials=args.n_trials
        )

        best_params = study.best_params
        best_score = study.best_value
        logger.info(f"最佳交叉验证分数 (F1-Macro): {best_score:.4f}")
        logger.info(f"找到的最佳超参数组合: {best_params}")

        # 4. 用 Optuna 找到的最佳参数，在全量数据上重训一个终极版模型
        logger.info("正在使用最佳超参数组合，训练最终模型...")
        win = best_params.pop('window_size')
        sig = best_params.pop('sigma')
        step_rate = best_params.pop('step_rate')
        actual_step = max(1, int(win * step_rate))

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

        # 使用最佳的 win, sig, step 重新生成特征
        X_final, y_final = build_features(raw_df, def_cfg["cols"], sig, win, actual_step)
        
        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X_final, y_final)

        # 5. 保存模型与测试所依赖的 Config 配置文件
        save_path = Path(args.output_dir) / f"{model_name}.joblib"
        joblib.dump(clf, save_path)
        
        final_configs[model_name] = {
            "model_path": str(save_path),
            "cols": def_cfg["cols"],
            "win": win,
            "sig": sig,
            "actual_step": actual_step, 
            "num_class": num_class,
            "new_to_old": new_to_old, # 测试脚本会依赖这个把模型输出还原成 1-6
            "cv_score": best_score,
            "best_params": best_params
        }
        logger.info(f"模型 {model_name} 训练完成并已保存至: {save_path}")

    # 保存总配置 JSON，供 test.py 读取
    config_path = Path(args.output_dir) / "model_configs.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(final_configs, f, indent=4)

    logger.info(f"\n所有模型的 Optuna 搜索与训练已完成！总耗时: {time.time() - total_start:.2f}s")

if __name__ == "__main__":
    main()

