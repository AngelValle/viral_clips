@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set CT2_VERBOSE=1

:: Ejecutamos a través de rtk para filtrar la salida y ahorrar tokens/espacio en consola
rtk streamlit run ui.py

pause
