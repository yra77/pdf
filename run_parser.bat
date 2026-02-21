@echo off
setlocal

if "%~2"=="" (
  echo Usage: run_parser.bat input.pdf output.csv [--ocr] [--profile-config bank_profiles.yaml --profile default]
  exit /b 1
)

python statement_parser.py %*
