@echo off
cd /d "%~dp0"
"%USERPROFILE%\.venvs\chat-safety-agent\Scripts\python.exe" -m streamlit run app.py
