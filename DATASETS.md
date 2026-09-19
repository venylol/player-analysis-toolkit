# Public dataset releases

Large public datasets and trained artifacts are published as GitHub Release
assets instead of being committed to Git. Each release contains a generated
`SHA256SUMS.txt` and `release-assets.json` manifest.

The curated release set contains:

- the 3,531-game bilateral training dataset;
- the 10,000-game source-only and OQ-profile retrospective datasets;
- the 10,000-game and 11,200-game profile-aware model-ready datasets;
- the public 2,339-player OQ profile snapshot;
- the final 11,200-game warm-start ensemble;
- the base TCN checkpoint;
- the Windows 9950X Egaroucid server handoff;
- the formal 600-game black-by-white directed source Reference;
- the Sentinel, estimated-Elo, Anscombe, and calibration derivatives of that
  formal Reference.

Generated smoke runs, superseded experiment snapshots, caches, logs, and
player-specific investigation artifacts are intentionally excluded. Public OQ
profiles and public game records may be included when they are part of a
documented training or Reference dataset; locally generated accusations,
investigation groupings, anomaly conclusions, and per-player reports are not.

The code and repository documentation are licensed under GNU GPL v3. Release
assets retain any source or third-party notices shipped inside their archives.

To rebuild the assets, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File tools/repository/build_release_assets.ps1 `
  -SourceRoot . `
  -OutputDirectory C:\path\to\release-assets `
  -ManifestPath tools/repository/public-release-layout.json `
  -BlockedRegex '<local-private-pattern>'
```
