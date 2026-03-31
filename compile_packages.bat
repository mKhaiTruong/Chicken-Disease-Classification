@echo off
echo Compiling requirements-dev.in...
pip-compile --verbose requirements-dev.in

echo.
echo Done! Run pip-sync to install.
pause