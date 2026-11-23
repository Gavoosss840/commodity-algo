@echo off
REM Daily run on Windows. Schedule it with Task Scheduler after the CBOT close
REM (e.g. 22:30 Paris time), with TWS or IB Gateway already running.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m commodity_algo run
git add output
git commit -m "Daily fiche %date%" >nul 2>&1
git push >nul 2>&1
