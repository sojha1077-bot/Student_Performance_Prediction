@echo off
REM Double-click this file on Windows to set up and start the dashboard.
cd /d "%~dp0"
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat
echo Installing libraries (first time takes a few minutes)...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
echo Starting the dashboard...
streamlit run app.py
pause
