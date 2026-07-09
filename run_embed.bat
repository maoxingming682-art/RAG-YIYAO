@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set HF_ENDPOINT=https://hf-mirror.com
set "ROOT=%~dp0"
if not exist "%ROOT%logs" mkdir "%ROOT%logs"
"%ROOT%venv\Scripts\python.exe" -u "%ROOT%src\02_chunk_embed.py" > "%ROOT%logs\embed_log.txt" 2>&1
