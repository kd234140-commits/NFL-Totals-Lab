from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

from v2_side_live_v23 import (
    build_team_features,
    build_qb_features,
    build_qbr_features,
    build_ngs_features,
    build_snap_features,
    build_depth_features,
    devig,
    ev,
    fair_american,
    norm_team,
    _val,
    _coverage_report,
)

try:
    from v2_side_live_v23 import build_free_injury_features as _build_injury_features
except Exception:
    from v2_side_live_v23 import build_espn_injury_features as _build_injury_features

try:
    from v2_side_live_v24 import resolve_head_referee, compute_referee_tracker, _clean_name
except Exception:
    resolve_head_referee = None
    compute_referee_tracker = None
    def _clean_name(x):
        return str(x or "").strip()


@st.cache_resource(show_spinner=False)
def load_v25_bundle(root_str: str):
    root = Path(root_str)
    bundle = joblib.load(root / "v2_5_models.joblib")
    meta = {}
    mp = root / "v2_5_metadata.json"
    if mp.exists():
        meta = json.loads(mp.read_text(encoding="utf-8"))
    return bundle, meta


def _build_feature_row(
    *, root: Path, schedule: pd.DataFrame, game: Any, prev_pbp: pd.DataFrame, cur_pbp: pd.DataFrame,
    home: str, away: str, season: int, week: int, market_total: float,
    over_odds: float, under_odds: float, home_spread: float, home_spread_odds: float,
    away_spread_odds: float, home_ml: float, away_ml: float, dome: float,
):
    bundle, meta = load_v25_bundle(str(root))
    features = list(bundle["features"])
    live = {c: np.nan for c in features}

    # Historical convention: +spread_line means HOME favored. Sportsbook UI uses -3 for home -3.
    spread_line = -float(home_spread)
    live.update({
        "season": float(season), "week": float(week),
        "away_rest": float(_val(game, "away_rest", 7.0)), "home_rest": float(_val(game, "home_rest", 7.0)),
        "away_moneyline": float(away_ml), "home_moneyline": float(home_ml),
        "spread_line": spread_line,
        "away_spread_odds": float(away_spread_odds), "home_spread_odds": float(home_spread_odds),
        "total_line": float(market_total), "under_odds": float(under_odds), "over_odds": float(over_odds),
        "div_game": float(_val(game, "div_game", 0.0)),
        "is_dome_or_closed": float(dome),
        "is_open_roof": float(1.0 - dome if str(_val(game, "roof", "")).lower() == "open" else 0.0),
    })

    mlh, mla, mlhold = devig(home_ml, away_ml)
    sph, spa, sphold = devig(home_spread_odds, away_spread_odds)
    ov, un, thold = devig(over_odds, under_odds)
    live.update({
        "market_ml_side1_devig": mlh, "market_ml_side2_devig": mla, "market_ml_hold": mlhold,
        "market_spread_side1_devig": sph, "market_spread_side2_devig": spa, "market_spread_hold": sphold,
        "market_total_side1_devig": ov, "market_total_side2_devig": un, "market_total_hold": thold,
        "market_home_points": (float(market_total) + spread_line) / 2.0,
        "market_away_points": (float(market_total) - spread_line) / 2.0,
        "rest_diff_home_minus_away": float(_val(game, "home_rest", 7.0)) - float(_val(game, "away_rest", 7.0)),
    })

    families = {"team": False, "qb": False, "qbr": False, "ngs": False, "snap": False, "depth": False, "injuries": False}

    try:
        z, ok = build_team_features(prev_pbp, cur_pbp, home, away, season, week)
        live.update({k: v for k, v in z.items() if k in live})
        families["team"] = bool((ok or {}).get("team"))
    except Exception:
        pass

    home_qb = _val(game, "home_qb_id", None)
    away_qb = _val(game, "away_qb_id", None)
    try:
        z, ok = build_qb_features(season, week, home_qb, away_qb)
        live.update({k: v for k, v in z.items() if k in live}); families["qb"] = bool(ok)
    except Exception:
        pass
    try:
        z, ok = build_qbr_features(season, week, home, away)
        live.update({k: v for k, v in z.items() if k in live}); families["qbr"] = bool(ok)
    except Exception:
        pass
    try:
        z, ok = build_ngs_features(season, week, home, away)
        live.update({k: v for k, v in z.items() if k in live}); families["ngs"] = bool(ok)
    except Exception:
        pass
    try:
        z, ok = build_snap_features(season, week, home, away)
        live.update({k: v for k, v in z.items() if k in live}); families["snap"] = bool(ok)
    except Exception:
        pass
    try:
        z, ok = build_depth_features(season, pd.Timestamp(_val(game, "gameday")), home, away)
        live.update({k: v for k, v in z.items() if k in live}); families["depth"] = bool(ok)
    except Exception:
        pass

    injury_meta = {"source": "none", "teams_found": []}
    try:
        injury_out = _build_injury_features(home, away)
        if isinstance(injury_out, tuple) and len(injury_out) >= 3:
            z, ok, injury_meta = injury_out[0], injury_out[1], injury_out[2]
        elif isinstance(injury_out, tuple) and len(injury_out) == 2:
            z, ok = injury_out; injury_meta = {"source": "live injury feed", "teams_found": []}
        else:
            z, ok = {}, False
        live.update({k: v for k, v in z.items() if k in live}); families["injuries"] = bool(ok)
    except Exception:
        pass

    # QB-change flags, matching the leakage-safe historical definition as closely as the live schedule permits.
    try:
        sched = schedule.copy()
        sched["gameday"] = pd.to_datetime(sched["gameday"], errors="coerce")
        cutoff = pd.Timestamp(_val(game, "gameday"))
        past = sched[sched["gameday"] < cutoff].copy()
        def previous_qb(team):
            rows = []
            if {"home_team", "home_qb_id"}.issubset(past.columns):
                h = past[past["home_team"].map(norm_team) == team][["gameday", "home_qb_id"]].rename(columns={"home_qb_id": "qb"}); rows.append(h)
            if {"away_team", "away_qb_id"}.issubset(past.columns):
                a = past[past["away_team"].map(norm_team) == team][["gameday", "away_qb_id"]].rename(columns={"away_qb_id": "qb"}); rows.append(a)
            if not rows:
                return None
            q = pd.concat(rows, ignore_index=True).dropna(subset=["qb"]).sort_values("gameday")
            return str(q.iloc[-1]["qb"]) if len(q) else None
        hp, ap = previous_qb(home), previous_qb(away)
        live["home_qb_change"] = float(hp is not None and home_qb is not None and str(hp) != str(home_qb)) if home_qb is not None else np.nan
        live["away_qb_change"] = float(ap is not None and away_qb is not None and str(ap) != str(away_qb)) if away_qb is not None else np.nan
    except Exception:
        live["home_qb_change"] = live.get("home_depth_qb1_change", np.nan)
        live["away_qb_change"] = live.get("away_depth_qb1_change", np.nan)

    X = pd.DataFrame([[live.get(c, np.nan) for c in features]], columns=features)
    coverage = _coverage_report(features, live, week)
    return bundle, meta, X, live, families, injury_meta, coverage, spread_line, mlh, home_qb, away_qb


def _calibrated_probs(bundle, margin: float, spread_line: float, market_total: float, market_home_p: float):
    sigma = max(6.0, float(bundle.get("sigma_margin", 12.75)))
    p_margin_win = float(norm.cdf(float(margin) / sigma))
    edge = float(margin) - float(spread_line)

    # Moneyline calibration needs market consensus. Fall back to margin distribution if ML prices are missing/invalid.
    p_home_win = p_margin_win
    try:
        pm = min(max(p_margin_win, .001), .999)
        mk = min(max(float(market_home_p), .001), .999)
        xm = np.array([[
            math.log(pm / (1 - pm)),
            math.log(mk / (1 - mk)),
            float(spread_line),
            float(market_total),
            edge,
        ]], dtype=float)
        if np.isfinite(xm).all():
            coef = np.asarray(bundle.get("ml_calibrator_coef", []), dtype=float)
            intercept = float(bundle.get("ml_calibrator_intercept", 0.0))
            if coef.size == xm.shape[1]:
                z = float(intercept + np.dot(xm[0], coef))
                p_home_win = float(1.0 / (1.0 + math.exp(-max(min(z, 35.0), -35.0))))
    except Exception:
        pass

    p_home_cover = float(norm.cdf(edge / sigma))
    try:
        xc = np.array([[edge]], dtype=float)
        coef = np.asarray(bundle.get("cover_calibrator_coef", []), dtype=float)
        intercept = float(bundle.get("cover_calibrator_intercept", 0.0))
        if coef.size == 1:
            z = float(intercept + xc[0, 0] * coef[0])
            p_home_cover = float(1.0 / (1.0 + math.exp(-max(min(z, 35.0), -35.0))))
    except Exception:
        pass

    p_home_win = min(max(p_home_win, 1e-6), 1 - 1e-6)
    p_home_cover = min(max(p_home_cover, 1e-6), 1 - 1e-6)
    return p_home_win, p_home_cover, edge, p_margin_win


def build_side_prediction_v25(*, referee_override: str | None = None, apply_referee: bool = True, **kwargs):
    """V2.5 challenger: LightGBM residual ensemble + historical OOF probability calibration.

    The underlying training set is 2018-2025 only. 2026 is the forward test.
    The model intentionally reuses the same live-generatable feature definitions as V2.2/V2.3.
    """
    root = Path(kwargs["root"])
    bundle, meta, X, live, families, injury_meta, cov, spread_line, market_home_p, home_qb, away_qb = _build_feature_row(**kwargs)

    arr = X.to_numpy(float)
    r1 = float(bundle["margin_l1"].predict(arr)[0])
    r2 = float(bundle["margin_l2"].predict(arr)[0])
    margin_l1 = float(spread_line) + r1
    margin_l2 = float(spread_line) + float(bundle.get("l2_residual_shrink", .75)) * r2
    base_margin = float(bundle.get("l1_weight", .75)) * margin_l1 + float(bundle.get("l2_weight", .25)) * margin_l2

    ref_assignment = {"referee": "", "source": "unavailable", "url": ""}
    ref_tracker = {"available": False, "reason": "Referee overlay disabled or unavailable."}
    ref_adj = 0.0
    if apply_referee and resolve_head_referee is not None and compute_referee_tracker is not None:
        try:
            schedule = kwargs["schedule"]; game = kwargs["game"]
            season = int(kwargs["season"]); week = int(kwargs["week"])
            away = str(kwargs["away"]); home = str(kwargs["home"])
            auto = resolve_head_referee(schedule=schedule, game=game, season=season, week=week, away=away, home=home) or ref_assignment
            chosen = _clean_name(referee_override) if referee_override else _clean_name(auto.get("referee", ""))
            ref_assignment = dict(auto)
            if referee_override and chosen:
                ref_assignment = {"referee": chosen, "source": "manual override", "url": auto.get("url", ""), "auto_referee": auto.get("referee", "")}
            ref_tracker = compute_referee_tracker(
                schedule=schedule, referee=chosen, game=game, season=season, week=week,
                away=away, home=home, home_spread=float(kwargs["home_spread"]),
            )
            if ref_tracker.get("available"):
                ref_adj = float(ref_tracker.get("adjustment_points", 0.0))
        except Exception as exc:
            ref_tracker = {"available": False, "reason": str(exc)}
            ref_adj = 0.0

    final_margin = base_margin + ref_adj
    p_home_win, p_home_cover, model_edge, p_margin_win = _calibrated_probs(
        bundle, final_margin, spread_line, float(kwargs["market_total"]), market_home_p
    )
    p_away_win, p_away_cover = 1.0 - p_home_win, 1.0 - p_home_cover

    home_ml = float(kwargs["home_ml"]); away_ml = float(kwargs["away_ml"])
    home_spread_odds = float(kwargs["home_spread_odds"]); away_spread_odds = float(kwargs["away_spread_odds"])

    return {
        "model_name": meta.get("model_name", bundle.get("model_name", "NFL V2.5 LightGBM Ensemble")),
        "status": meta.get("status", "Frozen 2026 challenger forward test"),
        "predicted_margin": final_margin,
        "base_predicted_margin": base_margin,
        "l1_margin": margin_l1,
        "l2_margin": margin_l2,
        "model_edge_vs_spread": model_edge,
        "p_margin_win_uncalibrated": p_margin_win,
        "p_home_win": p_home_win, "p_away_win": p_away_win,
        "p_home_cover": p_home_cover, "p_away_cover": p_away_cover,
        "fair_home_ml": fair_american(p_home_win), "fair_away_ml": fair_american(p_away_win),
        "fair_home_spread": fair_american(p_home_cover), "fair_away_spread": fair_american(p_away_cover),
        "home_ml_ev": ev(p_home_win, home_ml), "away_ml_ev": ev(p_away_win, away_ml),
        "home_spread_ev": ev(p_home_cover, home_spread_odds), "away_spread_ev": ev(p_away_cover, away_spread_odds),
        "coverage": float(cov["coverage"]),
        "eligible_coverage": float(cov["eligible_coverage"]),
        "selected_feature_count": len(bundle["features"]),
        "eligible_feature_count": int(cov["eligible_count"]),
        "missing_selected": list(cov["missing"]),
        "structural_missing": list(cov["structural_missing"]),
        "unexpected_missing": list(cov["unexpected_missing"]),
        "families": families,
        "injury_meta": injury_meta,
        "spread_line_model": float(spread_line),
        "home_qb_id": home_qb, "away_qb_id": away_qb,
        "referee_assignment": ref_assignment,
        "referee_tracker": ref_tracker,
        "referee_adjustment_points": ref_adj,
        "development_metrics": meta.get("walk_forward_2021_2025", {}),
        "architecture": meta.get("architecture", ""),
    }
