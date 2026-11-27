@echo off
pwsh -NoProfile -File "%~dp0start-windows-console.ps1" -CopyToken Console
if errorlevel 1 pause
