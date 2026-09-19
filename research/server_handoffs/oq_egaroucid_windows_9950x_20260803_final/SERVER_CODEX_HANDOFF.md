# Context handoff: Windows 9950X OQ/Egaroucid continuation

The user is moving the unfinished full-cohort computation to a Windows server with an
AMD Ryzen 9 9950X. The server expires at 2026-08-04 01:59 China Standard Time.

Your task is already automated by `run_oq_server_until_deadline.ps1`. Read `AGENTS.md`
and `README_SERVER.md`, ensure the extracted package root is under Desktop, and run the
single supervisor command. Monitor its logs using sleep-based intervals; do not start
parallel ad-hoc copies.

Initial dataset intent:

- 10,000 strict OQ normal five-minute games (`tcb=300000`);
- a partially complete `position_hints.csv` is supplied and must be resumed;
- existing keys must not be recomputed;
- target is level18 hint6 for every actual placement by both players;
- level2 hint1 nobook remains mandatory for every newly analyzed actual placement;
- pass rows remain continuity markers and receive no engine search.

The supervisor first finishes the current cohort. If it finishes with at least 45
minutes before the 01:44 hard stop, it invokes the supplied new-rule puller and the
all-position analyzer concurrently. The pull target remains 4,000 games whose two
players are both in the current supplied Elo list, using >=11-existing-game players
and one-source-game-per-round balancing. Candidate exhaustion is a valid early stop.

Hard deadlines are not discretionary:

- 01:44 China time: all pulls and engine calculations stopped.
- 01:49 China time: data and `DELIVERY_STATUS.json` on Desktop; all task processes
  stopped, including the foreground supervisor.

The package includes the exact Windows AVX512 AMD Egaroucid 7.8.1 executable,
resources, license, analyzer, puller, status tool, original game tables, leaderboard,
and existing hints. The supervisor uses four level18 workers × four threads on the
16-core 9950X, plus the separate level2 hint1 worker.

Do not train the TCN model here. After completion, return the whole Desktop package
directory or at minimum `input/games`, `input/hints`, `logs`, and
`DELIVERY_STATUS.json` to the workstation agent.
