import pandas as pd

# 读取CSV文件
input_file = 'data3.csv'
output_file = 'data3.csv'

# 读取数据
df = pd.read_csv(input_file)

# 从time列提取小时 - 使用字符串提取方式处理异常格式
# time列格式: 2021-04-01 00:00:00 或带有毫秒
# 提取小时部分 (HH:MM:SS 中的HH)
df['hour'] = df['time'].astype(str).str.extract(r'(\d{2}):\d{2}:\d{2}')[0].astype(int)

# 保存结果
df.to_csv(output_file, index=False)

print(f"处理完成！")
print(f"数据行数: {len(df)}")
print(f"前5行预览:")
print(df[['time', 'hour']].head())
