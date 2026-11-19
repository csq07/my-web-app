@echo off
cd /d "%~dp0"
start "" http://localhost:8000
python start-server.py 8000
if errorlevel 1 py start-server.py 8000
pause
