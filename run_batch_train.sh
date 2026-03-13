#!/bin/bash

# 批量执行模型训练脚本
# 按顺序执行多个训练脚本，执行完一个后再执行下一个

# 要执行的训练脚本列表（使用相对路径）
scripts=(
    "model_203_xgboost_p8p9f9h/model_203_xgboost_p8p9f9h_optuna.py"
    "model_204_xgboost_p8p9f9w/model_204_xgboost_p8p9f9w_optuna.py"
    "model_205_xgboost_p8p9f9wh/model_205_xgboost_p8p9f9wh_optuna.py"
    "model_206_xgboost_p8p9f9_p8/model_206_xgboost_p8p9f9_p8_optuna.py"
    "model_207_xgboost_p8p9f9_p9/model_207_xgboost_p8p9f9_p9_optuna.py"
    "model_208_xgboost_p8p9f9_p8_spe/model_208_xgboost_p8p9f9_p8_spe_optuna.py"
)

total=${#scripts[@]}
success=0
failed=0

echo "========================================"
echo "批量训练任务开始"
echo "任务数量: $total"
echo "========================================"

for ((i=0; i<total; i++)); do
    script="${scripts[$i]}"
    taskName=$(basename "$script")
    
    echo ""
    echo ">>> 进度: $((i+1))/$total - 正在执行: $taskName"
    
    # 执行 Python 脚本，使用默认参数
    python "$script"
    
    if [ $? -eq 0 ]; then
        echo ">>> $taskName 执行成功"
        ((success++))
    else
        echo ">>> $taskName 执行失败"
        ((failed++))
    fi
    
    echo ""
done

echo "========================================"
echo "批量训练任务完成"
echo "成功: $success/$total"
echo "失败: $failed/$total"
echo "========================================"

if [ $failed -gt 0 ]; then
    exit 1
else
    exit 0
fi
