@echo off
echo ========================================
echo  Quora Duplicate Question Detector
echo  Flask Web App Launcher
echo ========================================
echo.

REM Check if saved_models directory exists
if not exist "saved_models" (
    echo ERROR: Models not found!
    echo.
    echo Please run the model saving cell in complete_project.ipynb first.
    echo Look for the section: "Save Models for Deployment"
    echo.
    pause
    exit /b 1
)

echo Starting Flask server...
echo.
echo Open your browser and go to: http://127.0.0.1:5000
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

python app.py

pause
