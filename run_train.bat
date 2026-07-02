@echo off
cd /d %~dp0
call .venv\Scripts\activate
python scripts\train_model.py
pause
