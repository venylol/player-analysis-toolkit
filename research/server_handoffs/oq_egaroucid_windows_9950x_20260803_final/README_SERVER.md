# Windows 9950X server package

## Quick start

1. Upload the ZIP to the Windows server.
2. Extract the package folder directly under the server user's Desktop.
3. Open PowerShell in the extracted folder.
4. Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_oq_server_until_deadline.ps1
```

Python 3 must be available as `python.exe`. The Egaroucid executable and all required
resources/DLLs are already included.

The supervisor runs in the foreground, while worker processes are hidden. It polls
with `Start-Sleep`, stops all work at 01:44 China time, and writes
`DELIVERY_STATUS.json` before exiting. Because the extracted root is required to be
under Desktop, all continuously updated CSVs are already delivered to Desktop before
the 01:49 cutoff.

## Output locations

- Games and move records: `input/games/`
- Engine node results: `input/hints/position_hints.csv`
- Resume state: `input/hints/progress.json`
- Runtime logs: `logs/`
- Final status: `DELIVERY_STATUS.json`

Do not run the supervisor twice and do not manually start another Egaroucid process.
