# -*- coding: utf-8 -*-
import pandas as pd
import os

def process_workday_feature(input_path, output_path):
    print(f"正在处理: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"文件不存在: {input_path}，请检查路径。")
        return
        
    # 读取数据
    df = pd.read_csv(input_path)
    
    # 兼容列名寻找日期列 (你之前的代码中可能是 'time' 或 'date')
    date_col = 'date' if 'date' in df.columns else 'time'
    if date_col not in df.columns:
        print(f"错误: 在 {input_path} 中找不到 'date' 或 'time' 列。")
        return

    # 1. 提取时间并转换为标准 datetime 格式
    # apply(lambda x: x.split(".")[0]) 用于去除类似 2023-10-01 12:00:00.000 后面的毫秒
    temp_date = pd.to_datetime(df[date_col].astype(str).apply(lambda x: x.split(".")[0]))
    
    # 2. 判断是否为工作日
    # dt.dayofweek 返回 0-6（0是周一，4是周五，5是周六，6是周日）
    # < 5 即可筛选出周一至周五，.astype(int) 会将 True 转为 1，False 转为 0
    df['is_workday'] = (temp_date.dt.dayofweek < 5).astype(int)
    
    # 3. 保存为新的 CSV 文件
    df.to_csv(output_path, index=False)
    print(f"处理完成，新数据已保存至: {output_path}")
    print(f"前5行预览:\n{df[[date_col, 'is_workday']].head()}\n" + "-"*40)

if __name__ == "__main__":
    # 请根据你实际的文件路径进行调整
    # 假设你的数据在上一级的 data 目录中
    data3_input = "./data3_new.csv"
    data3_output = "./data3_workday.csv"
    
    data4_input = "./data4_new.csv"
    data4_output = "./data4_workday.csv"
    
    # 处理 data3
    process_workday_feature(data3_input, data3_output)
    
    # 处理 data4
    process_workday_feature(data4_input, data4_output)
    
    print("所有文件处理完毕！")
