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

from sklearn.model_selection import GroupKFold, TimeSeriesSplit, cross_val_score
from sklearn.metrics import f1_score
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================
# 1) 单模型：all_wh 预测所有类（多分类）
# ==========================================

MODEL_NAME = "all_wh"
MODEL_DEFINITION = {
    "cols": ["p8", "p9", "f9", "hour", "is_workday"],
    "objective": "multi:softprob",
    "is_binary": False
}

def r4(x):  # 保留 4 位
    return round(float(x), 4)

def setup_logger(output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("OptunaTrainLogger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(Path(output_dir) / "training_optuna.log", mode="w", encoding="utf-8")
        ch = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

# ==========================================
# 2) 数据读取与标签映射（与你当前训练一致）
# ==========================================

def load_dataset(path, anno_path, label_map):
    df = pd.read_csv(path)
    if "8p" in df.columns:
        df = df.rename(columns={"time": "date", "8p": "p8", "9p": "p9", "9f": "f9"})

    df = df.ffill()
    df["date"] = pd.to_datetime(df["date"].astype(str).apply(lambda x: x.split(".")[0]))
    df["day"] = df["date"].dt.date
    #.apply(lambda x: x.replace(year=2025))

    annodata = pd.read_excel(anno_path)
    annodata["time"] = pd.to_datetime(annodata["time"]).dt.date
    annodata["typea"] = annodata["type"].map(label_map)  # 原始 label_id（可能不连续）
    return df, annodata

def build_contiguous_class_mapping(annodata):
    labels = sorted({int(x) for x in annodata["typea"].dropna().unique()})
    all_labels = [0] + [lb for lb in labels if lb != 0]  # 0 背景
    old_to_new = {old: new for new, old in enumerate(all_labels)}
    new_to_old = {new: old for old, new in old_to_new.items()}
    num_class = len(all_labels)
    return old_to_new, new_to_old, num_class

def get_labeled_raw_data_all_classes(df, annodata, feature_cols, old_to_new):
    """
    逐天构建序列，并且输出一个 groups 向量（用于 GroupKFold，按天分组）
    """
    days = sorted(list(set(df["day"])))
    data_list = []
    group_list = []

    for gi, t in enumerate(days):
        sub = df[df["day"] == t][feature_cols].copy()
        sub["anno"] = 0
        sub.reset_index(drop=True, inplace=True)

        day_annos = annodata[annodata["time"] == t]
        for _, row in day_annos.iterrows():
            if pd.isna(row.get("typea")):
                continue
            s, e = int(row["start"]), int(row["end"])
            s, e = max(0, s), min(len(sub) - 1, e)
            old_label = int(row["typea"])
            if old_label not in old_to_new:
                continue
            new_label = old_to_new[old_label]
            if e >= s:
                sub.loc[s:e, "anno"] = new_label

        data_list.append(sub)
        group_list.append(np.full(len(sub), gi, dtype=int))  # 该天所有点同一 group id

    if not data_list:
        return pd.DataFrame(), None

    full_df = pd.concat(data_list, axis=0).reset_index(drop=True)
    full_df["anno"] = full_df["anno"].astype(int)
    groups = np.concatenate(group_list, axis=0)
    return full_df, groups

# ==========================================
# 3) 特征构建：返回 X, y, groups（窗口对齐后的 groups）
# ==========================================

def build_features_with_groups(raw_df, groups, feature_cols, sigma, window_size, step_size):
    """
    与你的 build_features 对齐：
    - label 用窗口末端
    - X 用 flattened + sliding_window_view
    - 下采样用 step_size
    同时把 groups 也对齐到 y 的位置（窗口末端的 group）
    """
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
    g_all = groups[window_size - 1:]  # 窗口末端对应的 group

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
# 4) Optuna 目标函数（贝叶斯/TPE）
# ==========================================

def objective(trial, raw_df, groups_raw, feature_cols, num_class, device):
    # ---- A) 特征参数 ----
    win = trial.suggest_int("window_size", 50, 400)
    sig = trial.suggest_int("sigma", 0, 80)
    step_rate = trial.suggest_float("step_rate", 0.02, 0.25)  # 连续，TPE
    actual_step = max(1, int(win * step_rate))

    X, y, g = build_features_with_groups(raw_df, groups_raw, feature_cols, sig, win, actual_step)
    if len(y) < 200:
        return 0.0

    # ---- B) XGBoost 参数（多分类）----
    params = {
        "objective": "multi:softprob",
        "num_class": num_class,

        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": r4(trial.suggest_float("learning_rate", 0.01, 0.3, log=True)),

        "subsample": r4(trial.suggest_float("subsample", 0.5, 1.0)),
        "colsample_bytree": r4(trial.suggest_float("colsample_bytree", 0.5, 1.0)),

        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": r4(trial.suggest_float("gamma", 0.0, 5.0)),

        "reg_alpha": r4(trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True)),
        "reg_lambda": r4(trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True)),

        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    # GPU 兼容（xgboost>=2: device 参数；<2: gpu_hist）
    if device == "cuda":
        try:
            major = float(xgb.__version__.split(".")[0])
            if major < 2:
                params["tree_method"] = "gpu_hist"
            else:
                params["device"] = "cuda"
        except:
            params["device"] = "cuda"

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
# 5) 主程序：调参 -> 用最优参数重训 -> 保存
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="../data/data3_workday.csv")
    parser.add_argument("--anno_path", type=str, default="../data/anno_data9.0_2021.xlsx")
    parser.add_argument("--label_map_path", type=str, default="../data/label_new.json")
    parser.add_argument("--output_dir", type=str, default="./saved_models_optuna")
    parser.add_argument("--n_trials", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    args, _ = parser.parse_known_args()

    logger = setup_logger(args.output_dir)
    logger.info(f"Using device: {args.device}")

    with open(args.label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    logger.info("Loading Training Data...")
    df_train, annodata = load_dataset(args.train_data, args.anno_path, label_map)

    old_to_new, new_to_old, num_class = build_contiguous_class_mapping(annodata)
    logger.info(f"Detected num_class={num_class} (including background=0)")

    raw_df, groups_raw = get_labeled_raw_data_all_classes(
        df_train, annodata, MODEL_DEFINITION["cols"], old_to_new
    )
    if raw_df.empty:
        logger.error("No training data produced; abort.")
        return

    logger.info(f"Raw labeled sequence length: {len(raw_df)}. Start Optuna...")

    tune_start = time.time()
    sampler = optuna.samplers.TPESampler(seed=42)  # TPE=贝叶斯
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective(trial, raw_df, groups_raw, MODEL_DEFINITION["cols"], num_class, args.device),
        n_trials=args.n_trials
    )
    tune_time = time.time() - tune_start

    best_params = study.best_params
    best_score = study.best_value

    logger.info(f"Optuna finished. Time={tune_time:.2f}s, best_macroF1={best_score:.4f}")
    logger.info("Best params:")
    for k, v in best_params.items():
        logger.info(f"  {k}: {v}")

    # ---- 用 best_params 重训全量模型 ----
    best_win = int(best_params["window_size"])
    best_sig = int(best_params["sigma"])
    best_step_rate = float(best_params["step_rate"])
    best_actual_step = max(1, int(best_win * best_step_rate))

    # 构建最终训练数据
    X_final, y_final, _ = build_features_with_groups(
        raw_df, groups_raw, MODEL_DEFINITION["cols"], best_sig, best_win, best_actual_step
    )

    # 组装最终 XGB 参数（去掉特征参数）
    xgb_params = {
        "objective": "multi:softprob",
        "num_class": num_class,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "n_estimators": int(best_params["n_estimators"]),
        "max_depth": int(best_params["max_depth"]),
        "learning_rate": float(best_params["learning_rate"]),
        "subsample": float(best_params["subsample"]),
        "colsample_bytree": float(best_params["colsample_bytree"]),
        "min_child_weight": int(best_params["min_child_weight"]),
        "gamma": float(best_params["gamma"]),
        "reg_alpha": float(best_params["reg_alpha"]),
        "reg_lambda": float(best_params["reg_lambda"]),
    }

    if args.device == "cuda":
        try:
            major = float(xgb.__version__.split(".")[0])
            if major < 2:
                xgb_params["tree_method"] = "gpu_hist"
            else:
                xgb_params["device"] = "cuda"
        except:
            xgb_params["device"] = "cuda"

    logger.info("Training final model with best params...")
    t0 = time.time()
    final_clf = xgb.XGBClassifier(**xgb_params)
    final_clf.fit(X_final, y_final)
    logger.info(f"Final training done in {time.time()-t0:.2f}s")

    # ---- 保存模型与配置 ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"{MODEL_NAME}_optuna.joblib"
    joblib.dump(final_clf, model_path)

    config = {
        MODEL_NAME: {
            "model_path": str(model_path),
            "cols": MODEL_DEFINITION["cols"],
            "win": best_win,
            "sig": best_sig,
            "actual_step": best_actual_step,
            "step_rate": r4(best_step_rate),
            "num_class": num_class,
            "old_to_new": old_to_new,
            "new_to_old": new_to_old,
            "best_params": {k: (r4(v) if isinstance(v, float) else int(v)) for k, v in best_params.items()},
            "cv_score_macro_f1": r4(best_score),
            "model_type": "XGBClassifier_Optuna_TPE"
        }
    }

    with open(out_dir / "model_configs.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    logger.info(f"Saved model to: {model_path}")
    logger.info(f"Saved configs to: {out_dir / 'model_configs.json'}")


if __name__ == "__main__":
    main()
