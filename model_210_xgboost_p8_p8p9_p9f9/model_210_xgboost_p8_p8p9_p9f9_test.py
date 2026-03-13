# -*- coding: utf-8 -*-
"""
test.py: 加载模型并在测试集上评估
"""
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

def setup_logger(output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("TestLogger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(ch)
    return logger

def load_dataset(path, anno_path, label_map, other_id):
    df = pd.read_csv(path)
    df = df[["time", "8p", "9p", "9f"]]
    df.columns = ["date", "p8", "p9", "f9"]
    df = df.ffill()
    df['date'] = pd.to_datetime(df['date'].astype(str).apply(lambda x: x.split(".")[0]))
    df['day'] = df['date'].dt.date.apply(lambda x: x.replace(year=2025))
    
    annodata = pd.read_excel(anno_path)
    annodata['time'] = pd.to_datetime(annodata['time']).dt.date
    annodata['typea'] = annodata['type'].map(label_map)
    
    # 清洗其他类 -> 0
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
    
    # Inference步长固定为特征数 (即 sliding step = 1 row)
    return sliding_window_view(flattened, window_shape=real_window_len)[::num_feat]


def adjust_predicts(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    predicted = predicted.copy()
    anomaly_state = False
    for i in range(len(actual)):
        flag = predicted[i]
        # 进入异常状态
        if actual[i] == predicted[i] != 0 and not anomaly_state:
            anomaly_state = True
            # 回填
            for j in range(i, -1, -1):
                if actual[j] == 0 or actual[j] != flag: break
                if predicted[j] == 0: predicted[j] = flag
            # 后填 (预判后续连续段)
            for j in range(i, len(actual)):
                if actual[j] == 0 or actual[j] != flag: break
                if predicted[j] == 0: predicted[j] = flag
        # 离开异常状态
        elif actual[i] == 0:
            anomaly_state = False
        
        if anomaly_state:
            predicted[i] = flag
    return predicted

'''
def adjust_predicts_old(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """
    点调整逻辑
    """
    predicted = predicted.copy()
    n = len(actual)
    i = 0
    while i < n:
        if actual[i] != 0: 
            j = i
            while j < n and actual[j] == actual[i]:
                j += 1
            window_pred = predicted[i:j]
            if np.any(window_pred == actual[i]):
                predicted[i:j] = actual[i]
            i = j
        else:
            i += 1
    return predicted
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", default="../data/data3.csv")
    parser.add_argument("--anno_path", default="../data/anno_data9.0.xlsx")
    parser.add_argument("--label_map", default="../data/label.json")
    parser.add_argument("--model_dir", default="./final_models", help="训练脚本输出的目录")
    args = parser.parse_args()
    
    logger = setup_logger(args.model_dir)

    total_start_time = time.time()

    # 1. 加载配置和模型
    load_start = time.time()
    meta_path = Path(args.model_dir) / "model_metadata.json"
    if not meta_path.exists():
        logger.error("Metadata file not found! Please run train.py first.")
        return

    with open(meta_path, 'r') as f:
        metadata = json.load(f)
        
    models = {}
    for m_type, meta in metadata.items():
        models[m_type] = joblib.load(meta["path"])
        logger.info(f"Loaded {m_type}")

    load_end = time.time()
    logger.info(f"Models loaded in {load_end - load_start:.2f} seconds.")

    # 2. 加载测试数据
    with open(args.label_map, 'r') as f:
        label_map = json.load(f)
    other_id = label_map.get("其他", 7)
    id_to_name = {v: k for k, v in label_map.items() if v != other_id}
    
    logger.info("Loading Test Data...")
    df_test, annodata = load_dataset(args.test_data, args.anno_path, label_map, other_id)
    
    # 3. 推理循环
    logger.info("Starting Inference...")
    inference_start = time.time()

    days = sorted(list(set(df_test['day'])))
    full_preds = []
    full_labels = []
    
    for t in tqdm(days, desc="Testing"):
        day_data = df_test[df_test['day'] == t].copy()
        day_data.reset_index(drop=True, inplace=True)
        N = len(day_data)
        
        # 获取真实标签 (Other已为0)
        labels = np.zeros(N, dtype=int)
        day_annos = annodata[annodata['time'] == t]
        for _, row in day_annos.iterrows():
            s, e, tp = int(row['start']), int(row['end']), int(row['typea'])
            s, e = max(0, s), min(N-1, e)
            if e >= s: labels[s:e+1] = tp
            
        # 初始化各模型预测向量
        preds_vec = {k: np.zeros(N, dtype=int) for k in ["p8", "p8p9", "p9f9"]}
        
        # 模型推理
        for m_type in ["p8", "p8p9", "p9f9"]:
            if m_type not in models: continue
            
            cfg = metadata[m_type]
            model = models[m_type]
            
            # 构建特征 (必须使用训练时的 sig 和 win)
            X = create_features_inference(day_data, cfg['cols'], cfg['sig'], cfg['win'])
            
            if X is not None:
                raw_p = model.predict(X)
                
                # 映射回原始 ID
                final_p = np.zeros_like(raw_p)
                if m_type == "p8":
                    # 0->0, 1->3, 2->4, 3->5
                    mapping = {0:0, 1:3, 2:4, 3:5}
                    final_p = np.vectorize(mapping.get)(raw_p)
                elif m_type == "p8p9":
                    # 1->1
                    final_p = np.where(raw_p==1, 1, 0)
                elif m_type == "p9f9":
                    # 1->2, 2->6
                    mapping = {0:0, 1:2, 2:6}
                    final_p = np.vectorize(mapping.get)(raw_p)
                
                # 补齐长度 (头部padding)
                pad = N - len(final_p)
                if pad > 0:
                    final_p = np.concatenate([np.zeros(pad, dtype=int), final_p])
                
                preds_vec[m_type] = final_p

        # --- 融合逻辑 ---
        p8_v = preds_vec["p8"]
        p8p9_v = preds_vec["p8p9"]
        p9f9_v = preds_vec["p9f9"]
        
        # 冲突处理
        # 1. p8p9(1类) 与 p9f9(2/6类) 冲突 -> 优先 p9f9
        # (通常 p9f9 更准，因为有流量特征)
        # 这里保留你的原有逻辑：(p8!=0) & (p8p9==1) & (p9f9==0) => p8p9=0 ? 
        # 原逻辑：
        mask_310 = (p8_v != 0) & (p8p9_v == 1) & (p9f9_v == 0)
        p8p9_v[mask_310] = 0
        
        mask_complex = (p8_v != 0) & (p8p9_v != 0) & (p9f9_v == 2)
        p8_v[mask_complex] = 0
        p8p9_v[mask_complex] = 0
        
        mask_016 = (p8_v == 0) & (p8p9_v == 1) & (p9f9_v != 0)
        p8p9_v[mask_016] = 0
        
        # 合并
        final_res = np.maximum(np.maximum(p8_v, p8p9_v), p9f9_v)
        
        full_preds.extend(final_res)
        full_labels.extend(labels)

    inference_end = time.time()
    inference_duration = inference_end - inference_start
    total_points = len(full_preds)
    fps = total_points / inference_duration if inference_duration > 0 else 0
    
    logger.info(f"Inference finished in {inference_duration:.2f} seconds.")
    logger.info(f"Processing Speed: {fps:.0f} points/second")
    total_end_time = time.time()
    # 4. 生成报告
    full_labels = np.array(full_labels)
    full_preds = np.array(full_preds)
    
    # 填充处理
    preds_adj = adjust_predicts(full_labels, full_preds)
    
    unique_lbls = sorted(list(set(full_labels) | set(full_preds)))
    # 移除 0 (背景) 和 7 (其他 - 如果还有残留)
    unique_lbls = [x for x in unique_lbls if x != 0 and x != other_id]
    
    target_names = [id_to_name.get(i, f"Type {i}") for i in unique_lbls]
    
    print("\n" + "="*50)
    print("FINAL TEST REPORT (CLEAN)")
    print("="*50)
    print(classification_report(
        full_labels, 
        preds_adj, 
        labels=unique_lbls, 
        target_names=target_names, 
        digits=4
    ))

    logger.info(f"All tasks completed in {total_end_time - total_start_time:.2f} seconds.")
    
    # ==========================================
    # 新增：保存混淆矩阵与报告到 Excel
    # ==========================================
    from sklearn.metrics import confusion_matrix
    import pandas as pd 

    # 1. 计算混淆矩阵 (使用与报告完全一致的 labels)
    cm = confusion_matrix(full_labels, preds_adj, labels=unique_lbls)
    
    # 2. 转换为 DataFrame 并添加标签
    # 行索引 = 真实标签 (True Label)，列索引 = 预测标签 (Predicted Label)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    
    # 3. 生成分类报告的 DataFrame (用于保存到 Excel 的第二个 Sheet)
    # output_dict=True 可以让 classification_report 返回字典方便转 DataFrame
    report_dict = classification_report(
        full_labels, 
        preds_adj, 
        labels=unique_lbls, 
        target_names=target_names, 
        digits=4,
        output_dict=True
    )
    # 转置 DataFrame，使得行是类别，列是指标 (Precision/Recall...)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df = report_df.round(4)
    # 4. 写入 Excel (保存到 args.model_dir 或当前目录)
    # 假设你之前定义了 args.model_dir，这里拼接路径
    # 如果没有定义 args，直接写 "model_evaluation_result.xlsx" 即可
    save_path = Path(args.model_dir) / "model_evaluation_result.xlsx" 

    try:
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            # Sheet 1: 混淆矩阵
            cm_df.to_excel(writer, sheet_name='Confusion Matrix')
            
            # Sheet 2: 分类报告 (F1, Precision, Recall)
            report_df.to_excel(writer, sheet_name='Classification Report')
            
        print(f"\n[Success] Confusion Matrix & Report saved to: {save_path}")
    except Exception as e:
        print(f"\n[Error] Failed to save Excel: {e}")

    '''
    preds_adj = adjust_predicts_old(full_labels, full_preds)
    unique_lbls = sorted(list(set(full_labels) | set(full_preds)))
    unique_lbls = [x for x in unique_lbls if x != 0 and x != other_id]
    target_names = [id_to_name.get(i, f"Type {i}") for i in unique_lbls]
    print("\n" + "="*50)
    print("FINAL TEST REPORT (CLEAN)")
    print("="*50)
    print(classification_report(
        full_labels, 
        preds_adj, 
        labels=unique_lbls, 
        target_names=target_names, 
        digits=4
    ))
    '''

if __name__ == "__main__":
    main()

