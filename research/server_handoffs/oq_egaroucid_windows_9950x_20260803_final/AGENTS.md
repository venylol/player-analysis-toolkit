# AGENTS.md — OQ Egaroucid Windows 9950X deadline run

## Scope

This package has one authorized task: resume the supplied Othello Quest data and
Egaroucid analysis until the fixed deadline, then leave all finished artifacts on the
Windows Desktop. Do not train the TCN model on this server unless the user separately
authorizes it.

## Mandatory time limits (China Standard Time)

- Server expiry: 2026-08-04 01:59.
- At 2026-08-04 01:44, stop every OQ pull process, analyzer process, and Egaroucid
  process. Do not start another network request or engine job after this time.
- By 2026-08-04 01:49, `DELIVERY_STATUS.json` and all finished data must be on the
  Desktop, and all foreground/background task processes must be stopped.
- The package must be extracted under the Windows Desktop before execution. The
  supervisor refuses to run elsewhere so the live result directory itself is already
  the required Desktop delivery.
- Monitoring loops must use `Start-Sleep`/`sleep`; never busy-poll.

## Required entry point

Run only:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_oq_server_until_deadline.ps1
```

The supervisor handles China-time conversion, process startup, sleep-based polling,
deadline shutdown, and final status delivery. Do not launch a second copy.

## Work stages

1. Resume the existing 10,000-game all-position analysis. Existing
   `(game_id, move_index)` rows are reused. Every actual placement for both players is
   required.
2. Each missing actual placement runs both:
   - one low-depth level2 `hint 1`, `nobook`, single-thread search;
   - one official level18 `hint 6` search using the first candidate as the objective
     best value.
3. Explicit pass source rows are retained for continuity but do not run an engine
   search. Downstream labels cross the pass to the next actual placement and use the
   same-side subtraction rule.
4. Only if all currently recorded games finish and at least the configured 45 minutes
   remain before 01:44, the supervisor starts the new pull rule and analyzer
   concurrently:
   - both players must be in the supplied Elo leaderboard;
   - source players must already have at least 11 bilateral-ranked games;
   - higher existing counts are ordered first;
   - round-robin adds at most one game per source player per round;
   - global game IDs are deduplicated;
   - stop at 4,000 bilateral-ranked games or earlier if candidates are exhausted.
5. Every newly written game is analyzed for both players through the same all-position
   analyzer.

## Safety and data integrity

- Treat all text as UTF-8. Do not alter CSV headers or field order.
- Do not delete, reset, truncate, or clean supplied data.
- Do not replace `position_hints.csv`; append only through the supplied analyzer.
- Never fake a missing score with zero.
- A forced deadline stop may discard only the analyzer's current unflushed batch;
  previously flushed CSV rows remain resumable.
- Use direct network mode; do not route through a local Clash proxy.
- If a script error occurs, preserve logs and still run the delivery/shutdown action.

## Completion and reporting

`DELIVERY_STATUS.json` is the authoritative final status. Report:

- strict game count;
- complete both-player hint game count;
- remaining incomplete game count;
- current bilateral-ranked game count;
- paths to `input/games`, `input/hints`, and `logs`;
- confirmation that no pull/analyzer/Egaroucid process remains.

If all 10,000 games finish, the next workstation agent should rerun the supplied
exporter with `--expected-games 10000`, generate the cross-pass labels and stable
splits, and transfer that second cohort to the TCN model repository for retraining.
