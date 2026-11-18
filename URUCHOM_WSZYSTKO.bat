@echo off
REM ============================================================================
REM  Kliknij dwukrotnie, aby uruchomic caly pipeline (siatka + strojenie + test).
REM  Szczegoly: run_all.py. Wyniki: results\, logi: results\logi\.
REM ============================================================================
chcp 65001 >nul
setlocal
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
title Praca magisterska - pelny przebieg

REM --- znajdz Pythona (najpierw z PATH, potem miniconda) ---
set "PY="
where python >nul 2>&1 && set "PY=python"
if not defined PY if exist "%USERPROFILE%\miniconda3\python.exe" set "PY=%USERPROFILE%\miniconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
if not defined PY (
    echo [BLAD] Nie znaleziono Pythona. Zainstaluj go albo dodaj do PATH.
    pause
    exit /b 1
)

REM --- sprawdz biblioteki zanim zaczniemy wielogodzinny przebieg ---
"%PY%" -c "import pandas, sklearn, xgboost, lightgbm, torch" 2>nul
if errorlevel 1 (
    echo [BLAD] Brakuje bibliotek. Uruchom w tym folderze:
    echo     pip install -r requirements.txt
    echo     pip install torch --index-url https://download.pytorch.org/whl/cu126
    pause
    exit /b 1
)

REM --- tryb: z argumentu (np. "URUCHOM_WSZYSTKO.bat wznow") albo z menu ---
set "MODE=%~1"
if not defined MODE (
    echo.
    echo  ==========================================================
    echo    PRACA MAGISTERSKA - pelny przebieg eksperymentow
    echo  ==========================================================
    echo    [1] OD NOWA  - stare wyniki przenoszone do results\archiwum\
    echo    [2] WZNOW    - dokoncz przerwany przebieg ^(policzone sa pomijane^)
    echo    [3] Anuluj
    echo.
    echo    Czas pelnego przebiegu: ok. 10-14 godzin.
    echo    Komputer nie zasnie w trakcie. Laptop: podlacz zasilacz.
    echo.
    choice /c 123 /n /m "  Wybierz 1, 2 lub 3: "
    if errorlevel 3 exit /b 0
    if errorlevel 2 (set "MODE=wznow") else (set "MODE=od_nowa")
)

if /i "%MODE%"=="od_nowa" (
    "%PY%" -u run_all.py --fresh
) else if /i "%MODE%"=="wznow" (
    "%PY%" -u run_all.py
) else if /i "%MODE%"=="test" (
    "%PY%" run_all.py --help
) else (
    echo Nieznany tryb "%MODE%". Dozwolone: od_nowa, wznow.
    pause
    exit /b 1
)

echo.
if errorlevel 1 (
    echo  [!] Zakonczono z bledem - sprawdz logi w results\logi\
) else (
    echo  Gotowe. Status etapow: results\logi\status_cpu.txt i status_gpu.txt
    echo  Wyniki obejrzysz w 03_eksperymenty.ipynb i 04_wyniki.ipynb ^(Run All^).
)
echo.
pause
