param(
    [string]$PythonExe = "python"
)

& $PythonExe -m pytest -q
exit $LASTEXITCODE

