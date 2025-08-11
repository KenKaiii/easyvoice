@echo off
REM Cross-platform installation script for EasyVoice CLI (Windows)
REM Works on Windows 10+ with Python installed

setlocal enabledelayedexpansion

echo 🎤 EasyVoice CLI Installation Script (Windows)
echo =============================================

REM Check if Python is installed
echo [INFO] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [SUCCESS] Python %PYTHON_VERSION% found

REM Check if pip is available
echo [INFO] Checking pip installation...
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] pip is not available
    echo Please ensure pip is installed with Python
    pause
    exit /b 1
)
echo [SUCCESS] pip is available

REM Install EasyVoice
echo [INFO] Installing EasyVoice CLI...
python -m pip install --user .
if %errorlevel% neq 0 (
    echo [ERROR] Installation failed
    pause
    exit /b 1
)
echo [SUCCESS] EasyVoice CLI installed successfully!

REM Optional audio dependencies
echo.
set /p "INSTALL_AUDIO=Do you want to install audio processing dependencies? (y/N): "
if /i "!INSTALL_AUDIO!"=="y" (
    echo [INFO] Installing audio dependencies (this may take a while)...
    python -m pip install --user ".[audio]"
    if !errorlevel! equ 0 (
        echo [SUCCESS] Audio dependencies installed!
    ) else (
        echo [WARNING] Audio dependencies installation failed (you can install them later)
    )
)

REM Verify installation
echo [INFO] Verifying installation...
where easyvoice >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] easyvoice command is available globally
    
    REM Test the command
    echo quit | easyvoice >nul 2>&1
    if !errorlevel! equ 0 (
        echo [SUCCESS] EasyVoice CLI is working correctly!
    ) else (
        echo [WARNING] Command installed but may have issues
    )
) else (
    echo [WARNING] easyvoice command not found in PATH
    
    REM Try to find where it was installed
    for /f %%i in ('python -m site --user-base') do set USER_BASE=%%i
    set USER_SCRIPTS=%USER_BASE%\Scripts
    
    if exist "%USER_SCRIPTS%\easyvoice.exe" (
        echo [WARNING] easyvoice is installed at: %USER_SCRIPTS%\easyvoice.exe
        echo [INFO] Add %USER_SCRIPTS% to your Windows PATH to use 'easyvoice' globally
        echo [INFO] To add to PATH:
        echo   1. Press Win+R, type 'sysdm.cpl', press Enter
        echo   2. Click 'Environment Variables'
        echo   3. Under User variables, select PATH and click Edit
        echo   4. Click New and add: %USER_SCRIPTS%
        echo   5. Click OK on all dialogs
    )
)

echo.
echo [SUCCESS] Installation complete!
echo.
echo 🚀 Quick Start:
echo   easyvoice          # Start interactive CLI
echo   python -m easyvoice    # Alternative way to run
echo.
echo 📚 Available commands:
echo   chat    - Start text conversation
echo   ask     - Ask a single question
echo   status  - Show system status
echo   help    - Show all commands
echo.
echo 🔧 Optional audio setup:
echo   pip install --user easyvoice[audio]  # Install audio dependencies
echo.
echo [INFO] For more information, run: easyvoice help

pause