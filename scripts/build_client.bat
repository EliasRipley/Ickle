@echo off
REM Build Ickle single-binary client with PyInstaller.
REM Uses ickle_client.spec as the single source of truth for hidden imports
REM (kept in sync with src/app.py's command table -- see the regenerate
REM command in the comment at the top of that file) instead of duplicating
REM a separate, driftable --hidden-import list here.
cd /d "%~dp0.."
rmdir /s /q dist\ickle_client 2>nul
rmdir /s /q build 2>nul

python -m PyInstaller --log-level WARN --clean ickle_client.spec

if %ERRORLEVEL% EQU 0 (
    echo Build complete: dist\ickle_client\ickle_client.exe
) else (
    echo Build failed
)
