@echo off
title TrafficAI Dashboard
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo [CANH BAO] Chua tim thay moi truong ao .venv!
    echo Vui long chay file "setup.bat" truoc de tu dong cai dat thu vien.
    pause
    exit /b 1
)

echo =====================================================================
echo  DANG KHOI DONG HE THONG TRAFFICAI DASHBOARD...
echo =====================================================================

rem Kiem tra port :8001
netstat -ano | findstr /C:":8001 " | findstr LISTENING >nul
if not errorlevel 1 (
    echo.
    echo [CANH BAO] Cong 8001 dang duoc su dung boi mot tien trinh khac!
    echo Vui long dong tien trinh dang chay hoac truy cap http://localhost:8001
    echo.
    pause
    exit /b 1
)

echo  Dang khoi tao he thong va nap cac mo hinh AI...
echo  Trinh duyet se tu dong mo ngay khi cac mo hinh AI san sang (200 OK)!
echo  Tai khoan mac dinh: admin / 123456
echo =====================================================================

rem Polling endpoint /api/health toi da 120 giay bang PowerShell voi try/catch an toan
start "" powershell -NoProfile -WindowStyle Hidden -Command "$u='http://localhost:8001/api/health'; for($i=0; $i -lt 120; $i++){ try { $r=Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop; if($r.StatusCode -eq 200){ Start-Process 'http://localhost:8001'; break } } catch { } Start-Sleep -Seconds 1 }"

".venv\Scripts\python.exe" web_app.py

pause
