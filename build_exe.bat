@echo off
REM Build script for creating ECG Monitor executable on Windows
REM Run: build_exe.bat

echo ========================================
echo Building ECG Monitor Executable
echo ========================================
echo.

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

echo.
echo Building executable...
echo.

REM Build using spec file
pyinstaller ECG_Monitor.spec

if errorlevel 1 (
    echo.
    echo Build failed! Check the error messages above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo Executable location: dist\ECG_Monitor.exe
echo.
echo You can now distribute the ECG_Monitor.exe file.
echo.
pause




