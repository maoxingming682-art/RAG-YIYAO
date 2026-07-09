@echo off
chcp 65001 >nul
title 药业RAG Demo - 一键启动
set "ROOT=%~dp0"
set "PY=%ROOT%venv\Scripts\python.exe"
set "NGROK=%LOCALAPPDATA%\ngrok\ngrok.exe"
echo ============================================
echo   药业智能咨询助手 - 一键启动
echo ============================================
echo.

echo [1/3] 启动Flask后端服务...
if not exist "%PY%" (
  echo   未找到虚拟环境: %PY%
  echo   请先创建 venv 并安装依赖。
  pause
  exit /b 1
)
start "Flask-5006" /min "%PY%" -u "%ROOT%run.py"
echo   Flask启动中（等待20秒加载embedding模型）...
timeout /t 20 /nobreak >nul
echo   打开浏览器: http://localhost:5006
start "" "http://localhost:5006"

echo.
echo [2/3] 启动ngrok公网隧道...
if exist "%NGROK%" (
  start "ngrok" /min "%NGROK%" http 5006 --log=stdout
  timeout /t 8 /nobreak >nul
) else (
  echo   未找到ngrok，跳过公网隧道。
)

echo.
echo [3/3] 获取公网地址...
powershell -Command "try { $t = Invoke-RestMethod -Uri 'http://localhost:4040/api/tunnels' -TimeoutSec 5; foreach($i in $t.tunnels){ Write-Host '公网地址:' $i.public_url } } catch { Write-Host 'ngrok还在启动中，稍等...' }"

echo.
echo ============================================
echo   启动完成！
echo   电脑访问: http://localhost:5006
echo   手机访问: 看上方的公网地址
echo ============================================
echo.
echo   注意：
echo   1. 电脑不能关机，关了服务就停
echo   2. ngrok免费版地址每次启动会变
echo   3. 给HR看时当天启动当天给地址
echo   4. 关闭服务：关掉这两个最小化窗口
echo.
pause
