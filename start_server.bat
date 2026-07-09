@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set HF_ENDPOINT=https://hf-mirror.com
set "ROOT=%~dp0"
"%ROOT%venv\Scripts\python.exe" -u "%ROOT%src\11_serve_finetuned.py"
