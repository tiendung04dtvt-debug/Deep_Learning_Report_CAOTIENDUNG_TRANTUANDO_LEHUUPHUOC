@echo off
title TrafficAI Setup
cd /d "%~dp0"

echo ======================================================
echo  DANG KHOI TAO MOI TRUONG PYTHON CHO TRAFFICAI
echo ======================================================

set "PY_CMD="
where python >nul 2>nul
if %errorlevel% equ 0 (
    set "PY_CMD=python"
) else (
    where py >nul 2>nul
    if %errorlevel% equ 0 (
        set "PY_CMD=py -3"
    ) else (
        echo [LOI] Khong tim thay Python!
        echo Vui long cai dat Python 3.10+ tu python.org va nho tich vao o "Add Python to PATH".
        pause
        exit /b 1
    )
)

echo Tim thay lenh Python: %PY_CMD%

if not exist .venv (
    echo [1/3] Dang tao moi truong ao .venv...
    %PY_CMD% -m venv .venv
) else (
    echo [1/3] Moi truong ao .venv da ton tai.
)

echo [2/3] Dang kich hoat moi truong ao va cai dat thu vien...
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -r requirements.txt

if exist requirements-ai.txt (
    pip install -r requirements-ai.txt
)

echo.
echo ======================================================
echo  CAI DAT HOAN TAT THANH CONG!
echo  Hay chay file "run.bat" de bat dau he thong.
echo ======================================================
pause
