# -*- coding: utf-8 -*-
"""
批量执行模型训练脚本
按顺序执行多个训练脚本，执行完一个后再执行下一个
"""

import subprocess
import sys
from pathlib import Path

# 获取脚本所在目录（项目根目录）
SCRIPT_DIR = Path(__file__).parent.resolve()

# 要执行的训练脚本列表（相对于项目根目录的路径）
TRAINING_TASKS = [
    "model_203_xgboost_p8p9f9h/model_203_xgboost_p8p9f9h_optuna.py",
    "model_204_xgboost_p8p9f9w/model_204_xgboost_p8p9f9w_optuna.py",
    "model_205_xgboost_p8p9f9wh/model_205_xgboost_p8p9f9wh_optuna.py",
    "model_206_xgboost_p8p9f9_p8/model_206_xgboost_p8p9f9_p8_optuna.py",
    "model_207_xgboost_p8p9f9_p9/model_207_xgboost_p8p9f9_p9_optuna.py",
    "model_208_xgboost_p8p9f9_p8_spe/model_208_xgboost_p8p9f9_p8_spe_optuna.py",
]


def main():
    total = len(TRAINING_TASKS)
    success = 0
    failed = 0

    print("========================================")
    print("批量训练任务开始")
    print(f"任务数量: {total}")
    print(f"项目根目录: {SCRIPT_DIR}")
    print("========================================")

    for i, task_path in enumerate(TRAINING_TASKS, 1):
        # 使用 Path 拼接完整路径
        script_path = SCRIPT_DIR / task_path
        task_name = script_path.name
        
        # 获取脚本所在目录，作为工作目录
        work_dir = script_path.parent

        print(f"\n>>> 进度: {i}/{total} - 正在执行: {task_name}")
        print(f"    脚本路径: {script_path}")
        print(f"    工作目录: {work_dir}")

        # 执行 Python 脚本，在脚本所在目录执行
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(work_dir),  # 在脚本所在目录下执行
            capture_output=False
        )

        if result.returncode == 0:
            print(f">>> {task_name} 执行成功")
            success += 1
        else:
            print(f">>> {task_name} 执行失败，返回码: {result.returncode}")
            failed += 1

        print()

    print("========================================")
    print("批量训练任务完成")
    print(f"成功: {success}/{total}")
    print(f"失败: {failed}/{total}")
    print("========================================")

    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
