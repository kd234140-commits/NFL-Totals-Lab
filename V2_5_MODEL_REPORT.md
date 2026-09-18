# NFL V2.5 Side Model

## Purpose
V2.5 is a new spread / moneyline challenger. It does **not** replace the frozen V2.2, V2.3 or V2.4 benchmarks. The 2026 season remains the real prospective test.

## Architecture
- Training seasons: 2018-2025 only
- 2,227 training games
- 631 live-generatable pregame features
- Margin target: actual home margin minus sportsbook spread
- 75% robust L1 LightGBM residual model
- 25% L2 LightGBM residual model, with that component's correction shrunk 25% toward the market
- Separate logistic calibration for moneyline win probability and spread cover probability
- Same small referee overlay used by V2.4 can move the final margin by at most +/-0.25 points

The feature set intentionally stays inside the families the deployed app can construct live: market/context, play-by-play team and matchup metrics, possession/drive data, QB weekly stats, ESPN QBR, Next Gen Stats, snap continuity, depth charts and injury counts.

## Historical development evidence
2021-2025 walk-forward predictions (1,424 games):
- V2.5 margin MAE: **9.245**
- Closing-spread baseline MAE: **9.762**
- Average improvement: **0.517 points/game**
- Bootstrap 95% interval for historical MAE improvement: approximately **+0.380 to +0.664 points/game**
- Historical ATS direction: **59.9%** across 1,391 non-push games

Per-season margin MAE vs market:
- 2021: 9.980 vs 10.667
- 2022: 8.300 vs 8.783
- 2023: 9.410 vs 9.984
- 2024: 9.223 vs 9.704
- 2025: 9.310 vs 9.670

A reference split with probability calibration fit on 2021-2022 and then checked on 2023-2025 produced moneyline Brier around **0.193** vs **0.210** for the de-vigged market and log loss around **0.568** vs **0.607**. Because the overall V2.5 architecture was iterated using 2021-2025 data, this is **not** being labeled a new untouched holdout.

## Important interpretation
These numbers are development evidence, not a promise of future betting performance. Model selection, feature design and calibration all used historical 2021-2025 results. The clean question is whether V2.5 continues to beat V2.2/V2.3/V2.4 and the market prospectively during 2026.

V2.5 deliberately leaves the totals model alone because the existing totals research has not beaten the market reliably.
