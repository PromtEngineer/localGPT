@echo off
title Python Script Menu
:menu
echo ====================================
echo          Python Script Menu
echo ====================================
echo 1. Run run_localGPT.py
echo 2. Run ingest.py
echo 3. Exit
echo ====================================
set /p choice=Please choose an option (1-3):

if "%choice%"=="1" goto run_localGPT
if "%choice%"=="2" goto run_ingest
if "%choice%"=="3" goto exit
echo Invalid choice, please select 1, 2, or 3.
goto menu

:run_localGPT
echo Running run_localGPT.py...
python run_localGPT.py
pause
goto menu

:run_ingest
echo Running ingest.py...
python ingest.py
pause
goto menu

:exit
echo Exiting...
pause
exit
