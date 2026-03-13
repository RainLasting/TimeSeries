@echo off
chcp 65001 >nul

:: 批量执行模型训练脚本
:: 按顺序执行多个训练脚本，执行完一个后再执行下一个

echo ========================================
echo 批量训练任务开始
echo ========================================

set success=0
set failed=0

:: 任务1: model_203
echo.
echo >>> 进度: 1/6 - 正在执行: model_203_xgboost_p8p9f9h_optuna.py
python model_203_xgboost_p8p9f9h\model_203_xgboost_p8p9f9h_optuna.py
if %errorlevel% equ 0 (
    echo >>> model_203 执行成功
    set /a success+=1
) else (
    echo >>> model_203 执行失败
    set /a failed+=1
)

:: 任务2: model_204
echo.
echo >>> 进度: 2/6 - 正在执行: model_204_xgboost_p8p9f9w_optuna.py
python model_204_xgboost_p8p9f9w\model_204_xgboost_p8p9f9w_optuna.py
if %errorlevel% equ 0 (
    echo >>> model_204 执行成功
    set /a success+=1
) else (
    echo >>> model_204 执行失败
    set /a failed+=1
)

:: 任务3: model_205
echo.
echo >>> 进度: 3/6 - 正在执行: model_205_xgboost_p8p9f9wh_optuna.py
python model_205_xgboost_p8p9f9wh\model_205_xgboost_p8p9f9wh_optuna.py
if %errorlevel% equ 0 (
    echo >>> model_205 执行成功
    set /a success+=1
) else (
    echo >>> model_205 执行失败
    set /a failed+=1
)

:: 任务4: model_206
echo.
echo >>> 进度: 4/6 - 正在执行: model_206_xgboost_p8p9f9_p8_optuna.py
python model_206_xgboost_p8p9f9_p8\model_206_xgboost_p8p9f9_p8_optuna.py
if %errorlevel% equ 0 (
    echo >>> model_206 执行成功
    set /a success+=1
) else (
    echo >>> model_206 执行失败
    set /a failed+=1
)

:: 任务5: model_207
echo.
echo >>> 进度: 5/6 - 正在执行: model_207_xgboost_p8p9f9_p9_optuna.py
python model_207_xgboost_p8p9f9_p9\model_207_xgboost_p8p9f9_p9_optuna.py
if %errorlevel% equ 0 (
    echo >>> model_207 执行成功
    set /a success+=1
) else (
    echo >>> model_207 执行失败
    set /a failed+=1
)

:: 任务6: model_208
echo.
echo >>> 进度: 6/6 - 正在执行: model_208_xgboost_p8p9f9_p8_spe_optuna.py
python model_208_xgboost_p8p9f9_p8_spe\model_208_xgboost_p8p9f9_p8_spe_optuna.py
if %errorlevel% equ 0 (
    echo >>> model_208 执行成功
    set /a success+=1
) else (
    echo >>> model_208 执行失败
    set /a failed+=1
)

echo ========================================
echo 批量训练任务完成
echo 成功: %success%/6
echo 失败: %failed%/6
echo ========================================

if %failed% gtr 0 (
    exit /b 1
) else (
    exit /b 0
)
