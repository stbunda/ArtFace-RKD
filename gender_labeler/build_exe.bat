@echo off
cd /d "%~dp0"
echo Installing PyInstaller in FaceRecognition environment...
call conda run -n FaceRecognition pip install pyinstaller -q

echo.
echo Building standalone executable...
call conda run -n FaceRecognition pyinstaller ^
    --onefile ^
    --windowed ^
    --name GenderLabeler ^
    --icon NONE ^
    gender_labeler.py

echo.
if exist dist\GenderLabeler.exe (
    echo Build successful!
    echo Executable: %~dp0dist\GenderLabeler.exe
    echo.
    echo You can share dist\GenderLabeler.exe with anyone.
    echo They just need to keep it next to the data\ folder,
    echo or configure the paths in the Settings dialog.
) else (
    echo Build FAILED - check the output above for errors.
)
pause
