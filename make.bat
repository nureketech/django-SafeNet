@echo off
rem Windows equivalent of the Makefile: make.bat test-stage N=0
setlocal
set "PY=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
"%PY%" "%~dp0tasks.py" %*
exit /b %ERRORLEVEL%
