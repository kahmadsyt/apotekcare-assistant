@echo off
cd /d %~dp0
echo ==========================================================
echo ApotekCare Assistant - Setup dan Run Streamlit
echo ==========================================================
IF NOT EXIST .venv (
    python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\train_model.py
streamlit run app\app.py
pause
