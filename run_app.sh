#!/usr/bin/env bash
# Mac / Linux: run with  bash run_app.sh
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi
source .venv/bin/activate
echo "Installing libraries (first time takes a few minutes)..."
python -m pip install --upgrade pip >/dev/null
pip install -r requirements.txt
echo "Starting the dashboard..."
streamlit run app.py
