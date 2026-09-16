$ErrorActionPreference = 'Stop'
# Run from the repository. Supply a native CPython 3.12+ executable if needed.
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'venv failed; use native Windows CPython 3.12+' }
}
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& '.\.venv\Scripts\python.exe' -m pip install --no-deps -e .
