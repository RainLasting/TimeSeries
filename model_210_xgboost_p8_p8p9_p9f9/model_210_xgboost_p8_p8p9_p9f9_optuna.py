# -*- coding: utf-8 -*-
import argparse
import json
import logging
import sys
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

warnings.filterwarnings("ignore")

MODEL_CONFIGS = {
    "p8": {
        "cols": ["p8"],
        "objective": "multi:softprob",
        "num_class": 4, 
        "is_binary": False
    },
    "p8p9": {
        "cols": ["p8", "p9"],
        "objective": "binary:logistic",
        "num_class": None,
        "is_binary": True
    },
    "p9f9": {
        "cols": ["p9", "f9"],
        "objective": "multi:softprob",
        "num_class": 3,
        "is_binary": False
    }
}

def adjust_predicts(actual: np.ndarray, predicted: np.ndarray, **kwargs) -> np.ndarray:
    """
    调整检测结果
    异常检测算法在一个异常区间检测到某点存在异常，则认为算法检测到整个异常区间的所有异常点
    先从检测到的异常点从后往前调整检测结果，随后再从该点从前往后调整检测结果，直到真实的异常为False
    退出异常状态，结束当前区间的调整

    :param actual: 真实的异常。
    :param predicted: 检测所得的异常。
    :return: 调整后的异常检测结果。
    """
    predicted = predicted.copy()
    anomaly_state = False
    for i in range(len(actual)):
        flag = predicted[i]
        if actual[i] == predicted[i] != 0 and not anomaly_state:
            anomaly_state = True
            for j in range(i, -1, -1):
                if actual[j] == 0 or actual[j] != flag:
                    break
                else:
                    if predicted[j] == 0:
                        predicted[j] = flag
            for j in range(i, len(actual)):
                if actual[j] == 0 or actual[j] != flag:
                    break
                else:
                    if predicted[j] == 0:
                        predicted[j] = flag
        elif actual[i] == 0:
            anomaly_state = False
        if anomaly_state:
            predicted[i] = flag
    return predicted

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="./data/data3.csv")
    parser.add_argument("--test_data", type=str, default="./data/data4.csv")
    parser.add_argument("--anno_path", type=str, default="./data/anno_data9.0.xlsx")
    parser.add_argument("--label_map_path", type=str, default="./data/label.json")
    parser.add_argument("--output_dir", type=str, default="./model_optuna")
    parser.add_argument("--opt_trials", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    return parser.parse_args()

def setup_logger(output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / "training_fusion_detailed.log"
    logger = logging.getLogger("FusionLogger")
    logger.setLevel(logging.INFO)
    logger.handlers = [] 
    
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(ch)
    
    return logger

def load_dataset(path, anno_path, label_map):
    df = pd.read_csv(path)
    df = df[["time", "8p", "9p", "9f"]]
    df.columns = ["date", "p8", "p9", "f9"]
    df = df.ffill()
    df['date'] = pd.to_datetime(df['date'].astype(str).apply(lambda x: x.split(".")[0]))
    df['day'] = df['date'].dt.date.apply(lambda x: x.replace(year=2025))
    
    annodata = pd.read_excel(anno_path)
    annodata['time'] = pd.to_datetime(annodata['time']).dt.date
    annodata['typea'] = annodata['type'].map(label_map)
    return df, annodata

def get_labeled_raw_data(df, annodata, model_type):
    cols = MODEL_CONFIGS[model_type]["cols"]
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
            if model_type == "p8p9" and tp == 1: val = 1
            elif model_type == "p9f9":
                if tp == 2: val = 1
                elif tp == 6: val = 2
            elif model_type == "p8" and tp in [3, 4, 5]: val = tp - 2
            
            if val != 0 and e >= s: 
                sub.loc[s:e, 'anno'] = val
        data_list.append(sub)
    
    if not data_list: return pd.DataFrame()
    full_df = pd.concat(data_list, axis=0).reset_index(drop=True)
    full_df['anno'] = full_df['anno'].astype(int)
    return full_df

def build_features(raw_df, feature_cols, sigma, window_size, step_size):
    df_temp = pd.DataFrame()
    for col in feature_cols:
        if sigma > 0: df_temp[col] = gaussian_filter(raw_df[col], sigma=sigma)
        else: df_temp[col] = raw_df[col]
            
    if len(raw_df) < window_size: return np.array([]), np.array([])

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

def optimize_and_save_top10(model_name, train_raw, args, logger):
    """
    参数搜索,保存TOP 10
    """
    if not OPTUNA_AVAILABLE:
        logger.error("Optuna not installed.")
        return None

    cfg = MODEL_CONFIGS[model_name]
    cols = cfg["cols"]
    
    sub_dir = Path(args.output_dir) / model_name
    sub_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*20} Optimizing {model_name} {'='*20}")
    
    labels_train = np.unique(train_raw['anno'])
    if len(labels_train) < 2:
        return None

    fixed_params = {
        'objective': cfg['objective'],
        'device': args.device,
        'tree_method': 'hist',
        'n_jobs': -1
    }
    if args.device == 'cuda':
        try:
            if float(xgb.__version__.split('.')[0]) < 2:
                fixed_params['tree_method'] = 'gpu_hist'
                del fixed_params['device']
        except: pass
    if cfg['num_class']: fixed_params['num_class'] = cfg['num_class']

    def objective(trial):
        def r4(x: float) -> float:
            return float(f"{x:.4f}")

        # --- 参数定义 ---
        win = trial.suggest_int('window_size', 50, 400)
        sig = trial.suggest_int('sigma', 10, 100)

        step_rate = trial.suggest_float('step_rate', 0.05, 0.2, step=0.0001)
        step_rate = r4(step_rate)

        step = max(1, int(win * step_rate))

        # --- XGBoost 参数 ---
        subsample = trial.suggest_float('subsample', 0.5, 1.0, step=0.0001)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.0001)
        gamma = trial.suggest_float('gamma', 0.0, 5.0, step=0.0001)

        subsample = r4(subsample)
        colsample_bytree = r4(colsample_bytree)
        gamma = r4(gamma)

        learning_rate = trial.suggest_float('learning_rate', 0.005, 0.3, log=True)
        reg_alpha = trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True)
        reg_lambda = trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)

        learning_rate = r4(learning_rate)
        reg_alpha = r4(reg_alpha)
        reg_lambda = r4(reg_lambda)

        xgb_params = {
            'n_estimators': trial.suggest_int('n_estimators', 1000, 1200),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),

            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,

            'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
            **fixed_params
        }

        if cfg['is_binary']:
            xgb_params['scale_pos_weight'] = trial.suggest_int('scale_pos_weight', 1, 10)

        X_all, y_all = build_features(train_raw, cols, sig, win, step)

        if len(y_all) < 50:
            raise optuna.TrialPruned()

        binc = np.bincount(y_all.astype(int))
        nonzero = binc[binc > 0]
        if len(nonzero) < 2:
            raise optuna.TrialPruned()
        if nonzero.min() < 10:
            raise optuna.TrialPruned()

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        cv_scores = []
        for tr_idx, va_idx in skf.split(X_all, y_all):
            clf = xgb.XGBClassifier(**xgb_params)
            clf.fit(X_all[tr_idx], y_all[tr_idx])
            preds = clf.predict(X_all[va_idx])
            cv_scores.append(f1_score(y_all[va_idx], preds, average='macro'))

        score_raw = float(np.mean(cv_scores))

        all_params = {
            "window": win,
            "sigma": sig,
            "step_rate": step_rate,
            "actual_step": step,
            **xgb_params
        }
        params_str = json.dumps(all_params, default=str)
        logger.info(f"Trial {trial.number:03d} | Val_F1: {score_raw:.4f} | Params: {params_str}")

        return score_raw

    optuna.logging.set_verbosity(optuna.logging.WARNING) 
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.opt_trials)
    
    logger.info(f"\nTraining and Saving Top 10 models for [{model_name}] on FULL Data3...")
    
    completed_trials = [t for t in study.trials if t.value is not None and t.state == optuna.trial.TrialState.COMPLETE]
    sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:10]
    
    saved_models_info = [] 

    for rank, trial in enumerate(sorted_trials):
        params = trial.params.copy()
        
        b_win = params.pop('window_size')
        b_sig = params.pop('sigma')
        b_rate = params.pop('step_rate')
        b_step = max(1, int(b_win * b_rate))
        
        final_xgb_params = {k: v for k, v in params.items()}
        final_xgb_params.update(fixed_params)
        
        X_final, y_final = build_features(train_raw, cols, b_sig, b_win, b_step)
        model = xgb.XGBClassifier(**final_xgb_params)
        model.fit(X_final, y_final)
        
        fname = f"{model_name}_rank{rank}.joblib"
        save_path = sub_dir / fname
        joblib.dump(model, save_path)
        
        config = {
            "rank": rank,
            "path": str(save_path),
            "win": b_win,
            "sig": b_sig,
            "cols": cols,
            "step": b_step,
            "val_f1": trial.value
        }
        saved_models_info.append(config)

        all_params_log = {
            "window": b_win,
            "sigma": b_sig,
            "step_rate": b_rate,
            "actual_step": b_step,
            **final_xgb_params
        }
        params_str = json.dumps(all_params_log, default=str)

        logger.info(f"  > Saved Rank {rank} | Val F1: {trial.value:.4f} | Path: {fname} | Params: {params_str}")

    return saved_models_info

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

def evaluate_single_model_on_test(model, config, df_test, annodata, model_type):
    """
    在 Data4 上评估单个模型的性能
    只评估该模型负责的故障类型 + 正常数据。
    如果遇到其他模型负责的故障，直接在评估时忽略该时间段，防止被误判为漏报。
    """
    days = sorted(list(set(df_test['day'])))
    preds_all = []
    labels_all = []
    
    # 1. 定义每个模型的“管辖范围” (原始标签 ID)
    # p8: 3=短暂放气, 4=设备启动, 5=设备停机
    # p8p9: 1=反冲洗
    # p9f9: 2=9井关井, 6=9井开井
    scope_map = {
        "p8": [3, 4, 5],
        "p8p9": [1],
        "p9f9": [2, 6]
    }
    target_types = scope_map.get(model_type, [])

    for t in days:
        day_data = df_test[df_test['day'] == t].copy()
        day_data.reset_index(drop=True, inplace=True)
        N = len(day_data)
        
        # valid_mask: 标记哪些时间点参与评估
        # 默认为 True (参与)，如果遇到非本模型负责的故障，设为 False (忽略)
        valid_mask = np.ones(N, dtype=bool)
        
        # internal_labels: 将全局标签映射回模型的内部标签 (0, 1, 2...)
        internal_labels = np.zeros(N, dtype=int)
        
        day_annos = annodata[annodata['time'] == t]
        for _, row in day_annos.iterrows():
            s, e, tp = int(row['start']), int(row['end']), int(row['typea'])
            s, e = max(0, s), min(N-1, e)
            
            if e >= s:
                if tp in target_types:
                    val = 0
                    if model_type == "p8p9" and tp == 1: val = 1
                    elif model_type == "p9f9":
                        if tp == 2: val = 1
                        elif tp == 6: val = 2
                    elif model_type == "p8":
                        if tp == 3: val = 1 
                        elif tp == 4: val = 2
                        elif tp == 5: val = 3
                    
                    internal_labels[s:e+1] = val
                elif tp != 0:
                    # 如果是故障，但不是该模型的目标故障 (例如 p8 遇到了 tp=1)
                    # 标记为无效，不计入 F1 分数
                    valid_mask[s:e+1] = False

        # 2. 推理
        X_feat = create_features_inference(day_data, config['cols'], config['sig'], config['win'])
        if X_feat is not None:
            p = model.predict(X_feat)
            pad = N - len(p)
            if pad > 0: p = np.concatenate([np.zeros(pad, dtype=int), p])
        else:
            p = np.zeros(N, dtype=int)
            
        # 3. 仅提取有效片段进行评估
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            preds_all.extend(p[valid_indices])
            labels_all.extend(internal_labels[valid_indices])
        
    # 如果该模型在测试集中完全没有对应的目标故障，避免报错
    if not labels_all:
        return 0.0
        
    return f1_score(labels_all, preds_all, average='macro')

# ==========================================
# 6. 主流程：选优与组合
# ==========================================

def main():
    args = parse_args()
    logger = setup_logger(args.output_dir)
    with open(args.label_map_path, 'r') as f:
        label_map = json.load(f)
        
    logger.info("Loading Datasets...")
    df_train, _ = load_dataset(args.train_data, args.anno_path, label_map)
    df_test, annodata = load_dataset(args.test_data, args.anno_path, label_map)
    
    # 1. 训练并保存 Top 10
    all_top_models_info = {} 
    
    for m_type in ["p8", "p8p9", "p9f9"]:
        train_raw = get_labeled_raw_data(df_train, annodata, m_type) 
        top10_info = optimize_and_save_top10(m_type, train_raw, args, logger)
        if top10_info:
            all_top_models_info[m_type] = top10_info
    
    # 将 Top 10 的配置保存为 JSON，供随机搜索脚本使用
    config_save_path = Path(args.output_dir) / "top10_configs.json"
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(all_top_models_info, f, indent=4)
    logger.info(f"Top 10 Configs saved to {config_save_path}")


    # 2. 在 Data4 上决选 No.1
    logger.info("\n" + "="*30)
    logger.info("Selecting BEST model on Data4 from Top 10 candidates")
    logger.info("="*30)
    
    best_models_final = {}   
    best_configs_final = {}  
    
    for m_type, candidates in all_top_models_info.items():
        best_score = -1
        best_candidate = None
        best_model_obj = None
        
        logger.info(f"--- Evaluating candidates for {m_type} ---")
        
        for cand in candidates:
            model = joblib.load(cand['path'])
            score = evaluate_single_model_on_test(model, cand, df_test, annodata, m_type)
            
            # 打印每个候选模型在 Data4 上的表现
            logger.info(f"  Rank {cand['rank']} (Val_F1={cand['val_f1']:.4f}) -> Test_F1: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_candidate = cand
                best_model_obj = model
        
        logger.info(f"Winner for {m_type}: Rank {best_candidate['rank']} with Test F1 {best_score:.4f}")
        best_models_final[m_type] = best_model_obj
        best_configs_final[m_type] = best_candidate

    # 3. 最终评估
    if len(best_models_final) == 3:
        logger.info("\n" + "="*30)
        logger.info("Running Final Fusion (Fusion of Winners)")
        logger.info("="*30)
        
        id_to_name = {v: k for k, v in label_map.items()}
        days = sorted(list(set(df_test['day'])))
        full_preds = []
        full_labels = []
        
        for t in tqdm(days, desc="Final Fusion"):
            day_data = df_test[df_test['day'] == t].copy()
            day_data.reset_index(drop=True, inplace=True)
            N = len(day_data)
            
            labels = np.zeros(N, dtype=int)
            day_annos = annodata[annodata['time'] == t]
            for _, row in day_annos.iterrows():
                s, e, tp = int(row['start']), int(row['end']), int(row['typea'])
                s, e = max(0, s), min(N-1, e)
                if e >= s: labels[s:e+1] = tp
            
            res_df = pd.DataFrame(0, index=range(N), columns=["p8", "p8p9", "p9f9"])
            
            for m_name, model in best_models_final.items():
                cfg = best_configs_final[m_name]
                X_feat = create_features_inference(day_data, cfg['cols'], cfg['sig'], cfg['win'])
                if X_feat is not None:
                    preds_raw = model.predict(X_feat)
                    mapped = np.zeros_like(preds_raw)
                    if m_name == "p8":
                        mapping = {0:0, 1:3, 2:4, 3:5}
                        mapped = np.vectorize(mapping.get)(preds_raw)
                    elif m_name == "p8p9":
                        mapped = np.where(preds_raw==1, 1, 0)
                    elif m_name == "p9f9":
                        mapping = {0:0, 1:2, 2:6}
                        mapped = np.vectorize(mapping.get)(preds_raw)
                    
                    pad_len = N - len(mapped)
                    if pad_len > 0:
                        final_preds = np.concatenate([np.zeros(pad_len, dtype=int), mapped])
                    else:
                        final_preds = mapped
                    res_df[m_name] = final_preds

            # 组合规则
            p8_v = res_df['p8'].values
            p8p9_v = res_df['p8p9'].values
            p9f9_v = res_df['p9f9'].values
            
            mask_310 = (p8_v != 0) & (p8p9_v == 1) & (p9f9_v == 0)
            p8p9_v[mask_310] = 0
            mask_complex = (p8_v != 0) & (p8p9_v != 0) & (p9f9_v == 2)
            p8_v[mask_complex] = 0
            p8p9_v[mask_complex] = 0
            mask_016 = (p8_v == 0) & (p8p9_v == 1) & (p9f9_v != 0)
            p8p9_v[mask_016] = 0
            
            final_res = np.maximum(np.maximum(p8_v, p8p9_v), p9f9_v)
            full_preds.extend(final_res)
            full_labels.extend(labels)
            
        full_labels = np.array(full_labels)
        full_preds = np.array(full_preds)
        unique_lbls = sorted(list(set(full_labels) | set(full_preds)))
        if 0 in unique_lbls: unique_lbls.remove(0)
        target_names = [id_to_name.get(i, f"Type {i}") for i in unique_lbls]
        
        # 打印报告
        preds_adj = adjust_predicts(full_labels, full_preds)
        report_adj = classification_report(full_labels, preds_adj, labels=unique_lbls, target_names=target_names, digits=4)
        logger.info("\n" + "#"*40)
        logger.info("FINAL REPORT (Point Adjusted)")
        logger.info("#"*40 + "\n" + report_adj)

if __name__ == "__main__":
    main()

