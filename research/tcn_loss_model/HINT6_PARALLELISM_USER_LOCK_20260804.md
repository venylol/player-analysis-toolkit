# Hint6 Parallelism — User-Locked Configuration

Effective date: 2026-08-04 (Asia/Shanghai)

This file records a direct user instruction and supersedes the earlier
single-Console conservative choice.

The formal frozen-10,000-game hint6 recomputation configuration is locked to:

- `workers = 12`;
- one independent Egaroucid Console process owned exclusively by each worker;
- `threads = 16` for every Console process;
- `hash level = 25` for every Console process;
- Egaroucid level 18, opening book enabled, and `-noautocacheclear` enabled;
- neither `-q` nor `-noboard` may be used;
- every request must retain the native Console board and full raw responses;
- the source remains the frozen 10,000-game CSV and must not be expanded.

Unless the user explicitly requests another change, no Agent may reduce,
increase, auto-tune, benchmark-select, or otherwise change the worker count,
thread count, hash level, engine level, book setting, or board-echo settings.
System load, runtime estimates, numeric search variability, or an Agent's own
conservative preference are not authorization to change this configuration.

The previous one-worker output under
`outputs/oq_safe_full_recompute_10000_20260804/hint6` is retained as immutable
interrupted-run evidence. It must not be deleted, overwritten, or presented as
the new twelve-worker formal result.

The authorized twelve-worker run uses the independent output root
`outputs/oq_safe_full_recompute_10000_hint6_w12_20260804`.

All twelve workers still use the safe transaction contract: a worker owns one
Console process, and `setboard + complete response read + hint + complete
response read` is atomic. Board provenance and legal-move validation remain
hard write gates.
