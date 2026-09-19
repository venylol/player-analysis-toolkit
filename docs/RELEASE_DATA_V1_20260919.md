# Public datasets and models — 2026-09-19

This release publishes the datasets and trained artifacts used by the initial
public version of Player Analysis Toolkit.

The assets include:

- bilateral 3,531-game and 10,000-game training datasets;
- 10,000-game and 11,200-game profile-aware model-ready datasets;
- the public 2,339-player OQ profile snapshot;
- the 11,200-game warm-start ensemble and base TCN checkpoint;
- the Windows 9950X Egaroucid handoff;
- the formal 600-game black-Elo-by-white-Elo directed source Reference;
- the matching Sentinel, estimated-Elo, Anscombe, and calibration derivatives.

`release-assets.json` records every source path, source size, compressed size,
and SHA-256 digest. `SHA256SUMS.txt` provides a compact verification list.

All assets are below 1 GiB. The largest asset is approximately 356 MiB.

Public OQ profiles and public game records are included where they form part of
a documented training or Reference dataset. Locally generated player-specific
investigation packages, reported-game groupings, anomaly conclusions, and
per-player reports are excluded.

Repository code and documentation are licensed under GNU GPL v3. Archives that
contain third-party components retain their bundled license and provenance
notices.
