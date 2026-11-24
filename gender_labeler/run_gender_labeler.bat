@echo off
cd /d "%~dp0"
if exist dist\GenderLabeler.exe (
    start "" dist\GenderLabeler.exe
) else (
    echo GenderLabeler.exe not found in dist\. Running from source via conda...
    call conda activate FaceRecognition
    python gender_labeler.py
)
