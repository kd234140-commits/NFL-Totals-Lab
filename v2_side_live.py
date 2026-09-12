from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import nflreadpy as nfl
except Exception:
    nfl = None

QBR_URL = "https://github.com/nflverse/nflverse-data/releases/download/espn_data/qbr_week_level.parquet"
ALPHA = 0.35
TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA", "JAC": "JAX", "WSH": "WAS"}
POSITION_GROUP = {
    "QB":"QB","RB":"RB","FB":"RB","WR":"SKILL","TE":"SKILL",
    "T":"OL","OT":"OL","G":"OL","OG":"OL","C":"OL","OL":"OL",
    "DE":"DL","DT":"DL","NT":"DL","DL":"DL","EDGE":"DL",
    "LB":"LB","ILB":"LB","OLB":"LB",
    "CB":"DB","S":"DB","FS":"DB","SS":"DB","DB":"DB",
    "K":"ST","P":"ST","LS":"ST",
}


def norm_team(x):
    if pd.isna(x):
        return x
    s = str(x).strip().upper()
    return TEAM_MAP.get(s, s)


def _to_pandas(x):
    if x is None:
        return pd.DataFrame()
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if hasattr(x, "to_pandas"):
        return x.to_pandas()
    return pd.DataFrame(x)


def _num(df, c, default=np.nan):
    if c in df.columns:
        return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _val(row, key, default=np.nan):
    try:
        v = row.get(key, default)
    except Exception:
        return default
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    return v


def implied_prob(odds):
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if not np.isfinite(o) or o == 0:
        return np.nan
    return 100.0 / (o + 100.0) if o > 0 else abs(o) / (abs(o) + 100.0)


def devig(odds_a, odds_b):
    a, b = implied_prob(odds_a), implied_prob(odds_b)
    if not np.isfinite(a) or not np.isfinite(b) or a + b <= 0:
        return np.nan, np.nan, np.nan
    hold = a + b - 1.0
    return a / (a + b), b / (a + b), hold


def fair_american(p):
    if p is None or not np.isfinite(p) or not (0 < p < 1):
        return np.nan
    return -100.0 * p / (1.0 - p) if p >= 0.5 else 100.0 * (1.0 - p) / p


def american_profit(odds):
    o = float(odds)
    return o / 100.0 if o > 0 else 100.0 / abs(o)


def ev(prob, odds):
    if prob is None or not np.isfinite(prob):
        return np.nan
    try:
        return float(prob) * american_profit(float(odds)) - (1.0 - float(prob))
    except Exception:
        return np.nan


def _roll_values(hist: pd.DataFrame, col: str, season: int):
    if hist.empty or col not in hist.columns:
        return {"r3": np.nan, "r5": np.nan, "r8": np.nan, "ewm": np.nan, "std": np.nan}
    s = pd.to_numeric(hist[col], errors="coerce")
    s_valid = s.dropna()
    cur = pd.to_numeric(hist.loc[hist["season"] == season, col], errors="coerce").dropna()
    return {
        "r3": float(s_valid.tail(3).mean()) if len(s_valid) else np.nan,
        "r5": float(s_valid.tail(5).mean()) if len(s_valid) else np.nan,
        "r8": float(s_valid.tail(8).mean()) if len(s_valid) else np.nan,
        "ewm": float(s_valid.ewm(alpha=ALPHA, adjust=False).mean().iloc[-1]) if len(s_valid) else np.nan,
        "std": float(cur.mean()) if len(cur) else np.nan,
    }


@st.cache_resource(show_spinner=False)
def load_side_bundle(root_str: str):
    root = Path(root_str)
    bundle = joblib.load(root / "v2_side_models.joblib")
    meta = {}
    mp = root / "v2_side_metadata.json"
    if mp.exists():
        meta = json.loads(mp.read_text(encoding="utf-8"))
    return bundle, meta


@st.cache_data(ttl=3600, show_spinner=False)
def load_player_stats_years(years: tuple[int, ...]):
    if nfl is None:
        return pd.DataFrame()
    parts = []
    for yr in years:
        try:
            d = _to_pandas(nfl.load_player_stats(int(yr), summary_level="week"))
            if len(d):
                parts.append(d)
        except Exception:
            pass
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_ngs_years(years: tuple[int, ...], kind: str):
    if nfl is None:
        return pd.DataFrame()
    parts = []
    for yr in years:
        try:
            d = _to_pandas(nfl.load_nextgen_stats(int(yr), stat_type=kind))
            if len(d):
                parts.append(d)
        except Exception:
            pass
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_snap_years(years: tuple[int, ...]):
    if nfl is None:
        return pd.DataFrame()
    parts = []
    for yr in years:
        try:
            d = _to_pandas(nfl.load_snap_counts(int(yr)))
            if len(d):
                parts.append(d)
        except Exception:
            pass
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def load_depth_current(season: int):
    if nfl is None:
        return pd.DataFrame()
    try:
        return _to_pandas(nfl.load_depth_charts(int(season)))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_qbr_years(years: tuple[int, ...]):
    try:
        d = pd.read_parquet(QBR_URL)
        d["season"] = pd.to_numeric(d["season"], errors="coerce")
        return d[d["season"].isin(list(years))].copy()
    except Exception:
        return pd.DataFrame()


def aggregate_team_game_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    if pbp is None or pbp.empty:
        return pd.DataFrame()
    p = pbp.copy()
    for c in ["posteam", "defteam", "td_team", "home_team", "away_team"]:
        if c in p.columns:
            p[c] = p[c].map(norm_team)

    no_play = _num(p, "no_play", 0).fillna(0).eq(1)
    valid = (~no_play) & p.get("posteam", pd.Series(index=p.index, dtype=object)).notna() & p.get("defteam", pd.Series(index=p.index, dtype=object)).notna()
    qb = _num(p, "qb_dropback", np.nan)
    if qb.notna().sum() == 0:
        qb = _num(p, "pass", 0)
    qb = qb.fillna(0).eq(1)
    pa = _num(p, "pass_attempt", np.nan)
    if pa.notna().sum() == 0:
        pa = _num(p, "pass", 0)
    pa = pa.fillna(0).eq(1)
    ra = _num(p, "rush_attempt", np.nan)
    if ra.notna().sum() == 0:
        ra = _num(p, "rush", 0)
    ra = ra.fillna(0).eq(1)
    scr = _num(p, "qb_scramble", 0).fillna(0).eq(1)
    kneel = _num(p, "qb_kneel", 0).fillna(0).eq(1)
    spike = _num(p, "qb_spike", 0).fillna(0).eq(1)
    play = valid & (qb | ra) & (~kneel) & (~spike)
    passplay = play & qb & (~scr)
    rushplay = play & ra & (~kneel)

    epa = _num(p, "epa")
    yards = _num(p, "yards_gained", 0)
    down = _num(p, "down")
    y100 = _num(p, "yardline_100")
    wp = _num(p, "wp")
    sd = _num(p, "score_differential")

    p["_play"] = play.astype(float)
    p["_pass"] = passplay.astype(float)
    p["_rush"] = rushplay.astype(float)
    p["_epa"] = epa.where(play)
    success = _num(p, "success")
    if success.notna().sum() == 0:
        success = (epa > 0).astype(float)
    p["_success"] = success.where(play)
    p["_pass_epa"] = epa.where(passplay)
    p["_rush_epa"] = epa.where(rushplay)
    p["_early_epa"] = epa.where(play & down.le(2))
    p["_late_epa"] = epa.where(play & down.ge(3))
    p["_neutral_epa"] = epa.where(play & wp.between(.2, .8) & sd.abs().le(8))
    p["_rz_epa"] = epa.where(play & y100.le(20))
    p["_ypp"] = yards.where(play)
    p["_expl20"] = (yards.ge(20) & play).astype(float).where(play)
    p["_pass_expl20"] = (yards.ge(20) & passplay).astype(float).where(passplay)
    p["_rush_expl10"] = (yards.ge(10) & rushplay).astype(float).where(rushplay)
    p["_turnover"] = ((_num(p, "interception", 0).fillna(0).eq(1)) | (_num(p, "fumble_lost", 0).fillna(0).eq(1))).astype(float).where(play)
    p["_sack"] = _num(p, "sack", 0).fillna(0).where(qb)
    p["_fd"] = _num(p, "first_down", 0).fillna(0).where(play)
    p["_shotgun"] = _num(p, "shotgun", 0).fillna(0).where(play)
    p["_nohuddle"] = _num(p, "no_huddle", 0).fillna(0).where(play)
    p["_cpoe"] = _num(p, "cpoe").where(pa)
    p["_air"] = _num(p, "air_yards").where(pa)
    p["_yac"] = _num(p, "yards_after_catch").where(pa)
    p["_passoe"] = (pa.astype(float) - _num(p, "xpass")).where(play & _num(p, "xpass").notna())
    p["_thirdconv"] = _num(p, "third_down_converted", 0).fillna(0).where(down.eq(3))
    p["_fourthconv"] = _num(p, "fourth_down_converted", 0).fillna(0).where(down.eq(4))

    td = ((_num(p, "touchdown", 0).fillna(0).eq(1)) & (p.get("td_team", pd.Series(index=p.index, dtype=object)) == p.get("posteam", pd.Series(index=p.index, dtype=object)))).astype(float) * 6
    fg = p.get("field_goal_result", pd.Series("", index=p.index)).astype(str).str.lower().eq("made").astype(float) * 3
    xp = p.get("extra_point_result", pd.Series("", index=p.index)).astype(str).str.lower().isin(["good", "made"]).astype(float)
    two = p.get("two_point_conv_result", pd.Series("", index=p.index)).astype(str).str.lower().isin(["success", "good"]).astype(float) * 2
    p["_pts"] = td + fg + xp + two
    p["_drive"] = p["fixed_drive"] if "fixed_drive" in p.columns and p["fixed_drive"].notna().any() else p.get("drive", pd.Series(np.nan, index=p.index))

    v = p[valid].copy()
    keys = [c for c in ["game_id", "posteam", "defteam", "season", "week"] if c in v.columns]
    spec = {
        "off_plays": ("_play", "sum"), "off_epa_per_play": ("_epa", "mean"), "off_epa_total": ("_epa", "sum"),
        "off_success_rate": ("_success", "mean"), "off_pass_epa": ("_pass_epa", "mean"), "off_rush_epa": ("_rush_epa", "mean"),
        "off_early_down_epa": ("_early_epa", "mean"), "off_late_down_epa": ("_late_epa", "mean"), "off_neutral_epa": ("_neutral_epa", "mean"),
        "off_redzone_epa": ("_rz_epa", "mean"), "off_yards_per_play": ("_ypp", "mean"), "off_explosive_rate_20": ("_expl20", "mean"),
        "off_explosive_pass_rate_20": ("_pass_expl20", "mean"), "off_explosive_rush_rate_10": ("_rush_expl10", "mean"),
        "off_turnover_rate": ("_turnover", "mean"), "off_sack_rate": ("_sack", "mean"), "off_first_down_rate": ("_fd", "mean"),
        "off_shotgun_rate": ("_shotgun", "mean"), "off_no_huddle_rate": ("_nohuddle", "mean"), "off_cpoe": ("_cpoe", "mean"),
        "off_air_yards": ("_air", "mean"), "off_yac": ("_yac", "mean"), "off_pass_oe": ("_passoe", "mean"), "off_pass_rate": ("_pass", "mean"),
        "off_third_down_conv": ("_thirdconv", "mean"), "off_fourth_down_conv": ("_fourthconv", "mean"),
    }
    named = {k: pd.NamedAgg(column=a, aggfunc=b) for k, (a, b) in spec.items()}
    tg = v.groupby(keys, dropna=False).agg(**named).reset_index()

    dr = v[v["_drive"].notna()].copy()
    if len(dr):
        dagg = {
            "drive_plays": pd.NamedAgg(column="_play", aggfunc="sum"),
            "drive_points": pd.NamedAgg(column="_pts", aggfunc="sum"),
            "drive_fds": pd.NamedAgg(column="_fd", aggfunc="sum"),
        }
        if "yardline_100" in dr.columns:
            dagg["min_y100"] = pd.NamedAgg(column="yardline_100", aggfunc="min")
            dagg["start_y100"] = pd.NamedAgg(column="yardline_100", aggfunc="first")
        d = dr.groupby(["game_id", "posteam", "_drive"], dropna=False).agg(**dagg).reset_index()
        d["score_drive"] = (d["drive_points"] > 0).astype(float)
        d["rz_drive"] = (pd.to_numeric(d.get("min_y100"), errors="coerce") <= 20).astype(float)
        d["three_out"] = ((d["drive_fds"].fillna(0) <= 0) & (d["drive_plays"].fillna(0) <= 3) & (d["drive_points"].fillna(0) <= 0)).astype(float)
        d["start_fp"] = 100 - pd.to_numeric(d.get("start_y100"), errors="coerce")
        dg = d.groupby(["game_id", "posteam"]).agg(
            off_drives=("_drive", "nunique"), off_points_per_drive=("drive_points", "mean"),
            off_drive_score_rate=("score_drive", "mean"), off_redzone_drive_rate=("rz_drive", "mean"),
            off_three_and_out_rate=("three_out", "mean"), off_start_field_position=("start_fp", "mean"),
        ).reset_index()
        tg = tg.merge(dg, on=["game_id", "posteam"], how="left")

    tg = tg.rename(columns={"posteam": "team", "defteam": "opponent"})
    tg["team"] = tg["team"].map(norm_team)
    tg["opponent"] = tg["opponent"].map(norm_team)
    off_cols = [c for c in tg.columns if c.startswith("off_")]
    opp = tg[["game_id", "team"] + off_cols].copy().rename(columns={"team": "opponent", **{c: "def_" + c[4:] + "_allowed" for c in off_cols}})
    return tg.merge(opp, on=["game_id", "opponent"], how="left")


def build_team_features(prev_pbp: pd.DataFrame, cur_pbp: pd.DataFrame, home: str, away: str, season: int, week: int):
    parts = []
    for p in [prev_pbp, cur_pbp]:
        x = aggregate_team_game_metrics(p)
        if len(x):
            parts.append(x)
    if not parts:
        return {}, {"team": False}
    tg = pd.concat(parts, ignore_index=True, sort=False)
    tg["season"] = pd.to_numeric(tg["season"], errors="coerce")
    tg["week"] = pd.to_numeric(tg["week"], errors="coerce")
    tg = tg[(tg["season"] < season) | ((tg["season"] == season) & (tg["week"] < week))].copy()
    tg = tg.sort_values(["season", "week", "game_id"])

    feat = {}
    snapshots = {}
    rolling_metrics = [
        "off_drives", "off_points_per_drive", "off_drive_score_rate", "off_three_and_out_rate", "off_start_field_position",
        "def_drives_allowed", "def_points_per_drive_allowed", "def_drive_score_rate_allowed", "def_three_and_out_rate_allowed", "def_start_field_position_allowed",
        "off_epa_per_play", "off_pass_epa", "off_rush_epa", "off_success_rate", "off_explosive_rate_20", "off_plays", "off_turnover_rate", "off_sack_rate", "off_pass_oe",
        "def_epa_per_play_allowed", "def_pass_epa_allowed", "def_rush_epa_allowed", "def_success_rate_allowed", "def_explosive_rate_20_allowed", "def_plays_allowed", "def_turnover_rate_allowed", "def_sack_rate_allowed", "def_pass_oe_allowed",
    ]
    prior_metrics = [
        "off_plays", "off_epa_per_play", "off_success_rate", "off_pass_epa", "off_rush_epa", "off_explosive_rate_20", "off_turnover_rate", "off_sack_rate", "off_drives", "off_points_per_drive",
        "def_plays_allowed", "def_epa_per_play_allowed", "def_success_rate_allowed", "def_pass_epa_allowed", "def_rush_epa_allowed", "def_explosive_rate_20_allowed", "def_turnover_rate_allowed", "def_sack_rate_allowed", "def_drives_allowed", "def_points_per_drive_allowed",
    ]

    for side, team in [("home", home), ("away", away)]:
        hist = tg[tg["team"] == team].copy()
        snap = {}
        for m in rolling_metrics:
            rv = _roll_values(hist, m, season)
            for suf in ["r5", "ewm", "std"]:
                snap[f"{m}_{suf}"] = rv[suf]
        snapshots[side] = snap

        # Explicit team pre features in frozen model.
        for m in [
            "off_drives", "off_points_per_drive", "off_drive_score_rate", "off_three_and_out_rate", "off_start_field_position",
            "def_drives_allowed", "def_points_per_drive_allowed", "def_drive_score_rate_allowed", "def_three_and_out_rate_allowed", "def_start_field_position_allowed",
        ]:
            for suf in ["r5", "ewm", "std"]:
                feat[f"{side}_pre_{m}_{suf}"] = snap.get(f"{m}_{suf}", np.nan)

        prior = hist[hist["season"] == season - 1]
        for m in prior_metrics:
            feat[f"{side}_prior_season_{m}"] = float(pd.to_numeric(prior.get(m), errors="coerce").mean()) if len(prior) and m in prior.columns else np.nan

    map_pairs = [
        ("epa", "off_epa_per_play", "def_epa_per_play_allowed"),
        ("pass_epa", "off_pass_epa", "def_pass_epa_allowed"),
        ("rush_epa", "off_rush_epa", "def_rush_epa_allowed"),
        ("success", "off_success_rate", "def_success_rate_allowed"),
        ("explosive", "off_explosive_rate_20", "def_explosive_rate_20_allowed"),
        ("plays", "off_plays", "def_plays_allowed"),
        ("ppd", "off_points_per_drive", "def_points_per_drive_allowed"),
        ("turnover", "off_turnover_rate", "def_turnover_rate_allowed"),
        ("sack", "off_sack_rate", "def_sack_rate_allowed"),
        ("passoe", "off_pass_oe", "def_pass_oe_allowed"),
    ]
    for suf in ["r5", "ewm", "std"]:
        for short, om, dm in map_pairs:
            ho = snapshots["home"].get(f"{om}_{suf}", np.nan)
            hd = snapshots["home"].get(f"{dm}_{suf}", np.nan)
            ao = snapshots["away"].get(f"{om}_{suf}", np.nan)
            ad = snapshots["away"].get(f"{dm}_{suf}", np.nan)
            mh = np.nanmean([ho, ad]) if np.isfinite(ho) or np.isfinite(ad) else np.nan
            ma = np.nanmean([ao, hd]) if np.isfinite(ao) or np.isfinite(hd) else np.nan
            feat[f"match_home_{short}_{suf}"] = mh
            feat[f"match_away_{short}_{suf}"] = ma
            feat[f"match_sum_{short}_{suf}"] = mh + ma if np.isfinite(mh) and np.isfinite(ma) else np.nan
            feat[f"match_diff_{short}_{suf}"] = mh - ma if np.isfinite(mh) and np.isfinite(ma) else np.nan

    # Possession x PPD features.
    for suf in ["r5", "ewm", "std"]:
        hd = np.nanmean([snapshots["home"].get(f"off_drives_{suf}", np.nan), snapshots["away"].get(f"def_drives_allowed_{suf}", np.nan)])
        ad = np.nanmean([snapshots["away"].get(f"off_drives_{suf}", np.nan), snapshots["home"].get(f"def_drives_allowed_{suf}", np.nan)])
        hppd = np.nanmean([snapshots["home"].get(f"off_points_per_drive_{suf}", np.nan), snapshots["away"].get(f"def_points_per_drive_allowed_{suf}", np.nan)])
        appd = np.nanmean([snapshots["away"].get(f"off_points_per_drive_{suf}", np.nan), snapshots["home"].get(f"def_points_per_drive_allowed_{suf}", np.nan)])
        feat[f"poss_home_drives_{suf}"] = hd
        feat[f"poss_away_drives_{suf}"] = ad
        feat[f"poss_home_ppd_{suf}"] = hppd
        feat[f"poss_away_ppd_{suf}"] = appd
        feat[f"poss_home_points_{suf}"] = hd * hppd if np.isfinite(hd) and np.isfinite(hppd) else np.nan
        feat[f"poss_away_points_{suf}"] = ad * appd if np.isfinite(ad) and np.isfinite(appd) else np.nan
        feat[f"poss_total_{suf}"] = feat[f"poss_home_points_{suf}"] + feat[f"poss_away_points_{suf}"] if np.isfinite(feat[f"poss_home_points_{suf}"]) and np.isfinite(feat[f"poss_away_points_{suf}"]) else np.nan
        feat[f"poss_margin_{suf}"] = feat[f"poss_home_points_{suf}"] - feat[f"poss_away_points_{suf}"] if np.isfinite(feat[f"poss_home_points_{suf}"]) and np.isfinite(feat[f"poss_away_points_{suf}"]) else np.nan

    return feat, {"team": True}


def build_qb_features(season: int, week: int, home_qb_id, away_qb_id):
    years = tuple(sorted(set([season - 2, season - 1, season])))
    d = load_player_stats_years(years)
    if d.empty or "player_id" not in d.columns:
        return {}, False
    if "team" in d.columns:
        d["team"] = d["team"].map(norm_team)
    d["season"] = pd.to_numeric(d["season"], errors="coerce")
    d["week"] = pd.to_numeric(d["week"], errors="coerce")
    d["player_id"] = d["player_id"].astype(str)
    keep = [c for c in [
        "passing_epa","passing_cpoe","passing_yards","passing_tds","passing_interceptions","sacks_suffered",
        "attempts","completions","carries","rushing_yards","rushing_tds","rushing_epa","passing_air_yards","passing_yards_after_catch"
    ] if c in d.columns]
    if not keep:
        return {}, False
    q = d.groupby(["player_id","season","week"], dropna=False)[keep].sum(min_count=1).reset_index()
    att = _num(q, "attempts", 0).fillna(0); sacks = _num(q, "sacks_suffered", 0).fillna(0); car = _num(q, "carries", 0).fillna(0)
    q["qb_dropbacks_est"] = att + sacks
    q["qb_total_plays_est"] = att + sacks + car
    q["qb_pass_epa_per_dropback"] = _num(q, "passing_epa") / q["qb_dropbacks_est"].replace(0, np.nan)
    q["qb_rush_epa_per_carry"] = _num(q, "rushing_epa") / car.replace(0, np.nan)
    q["qb_int_rate"] = _num(q, "passing_interceptions") / att.replace(0, np.nan)
    metrics = keep + ["qb_dropbacks_est","qb_total_plays_est","qb_pass_epa_per_dropback","qb_rush_epa_per_carry","qb_int_rate"]
    out = {}
    got = False
    for side, qid in [("home", home_qb_id), ("away", away_qb_id)]:
        if qid is None or pd.isna(qid):
            continue
        h = q[q["player_id"] == str(qid)].copy()
        h = h[(h["season"] < season) | ((h["season"] == season) & (h["week"] < week))].sort_values(["season","week"])
        if h.empty:
            continue
        got = True
        for m in metrics:
            rv = _roll_values(h, m, season)
            for suf in ["r5","ewm","std"]:
                out[f"{side}_qbpre_{m}_{suf}"] = rv[suf]
    return out, got


def build_qbr_features(season: int, week: int, home: str, away: str):
    raw = load_qbr_years(tuple(sorted(set([season - 1, season]))))
    if raw.empty:
        return {}, False
    d = raw.copy()
    if "team_abb" not in d.columns:
        return {}, False
    d["team"] = d["team_abb"].map(norm_team)
    d["week"] = pd.to_numeric(d.get("game_week", "").astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    metrics = [c for c in ["qbr_total","pts_added","epa_total","pass","run","penalty","qbr_raw","sack"] if c in d.columns]
    rows=[]
    for k,g in d.groupby(["season","week","team"],dropna=False):
        r={"season":k[0],"week":k[1],"team":k[2]}
        w = pd.to_numeric(g.get("qb_plays"), errors="coerce") if "qb_plays" in g.columns else None
        for c in metrics:
            x = pd.to_numeric(g[c], errors="coerce")
            if w is not None:
                ok=x.notna()&w.notna()&w.gt(0)
                r[f"qbr_{c}"] = float(np.average(x[ok], weights=w[ok])) if ok.any() else float(x.mean())
            else:
                r[f"qbr_{c}"] = float(x.mean())
        r["qbr_qb_plays"] = float(w.sum()) if w is not None else np.nan
        rows.append(r)
    t=pd.DataFrame(rows)
    out={}; got=False
    for side,team in [("home",home),("away",away)]:
        h=t[t.team==team].copy()
        h=h[(h.season<season)|((h.season==season)&(h.week<week))].sort_values(["season","week"])
        if h.empty: continue
        got=True
        for m in [c for c in t.columns if c.startswith("qbr_")]:
            rv=_roll_values(h,m,season)
            for suf in ["r5","ewm","std"]:
                out[f"{side}_qbr_pre_{m}_{suf}"]=rv[suf]
    return out, got


def _weighted_mean(g, value, weight):
    x=pd.to_numeric(g[value],errors="coerce")
    w=pd.to_numeric(g[weight],errors="coerce").fillna(0) if weight in g.columns else pd.Series(1.0,index=g.index)
    ok=x.notna()&w.gt(0)
    return float(np.average(x[ok],weights=w[ok])) if ok.any() else float(x.mean())


def build_ngs_features(season:int, week:int, home:str, away:str):
    years=tuple(sorted(set([season-1,season])))
    out={}; any_got=False
    configs={
        "passing":("attempts",["avg_time_to_throw","avg_completed_air_yards","avg_intended_air_yards","aggressiveness","avg_air_yards_to_sticks","completion_percentage_above_expectation","passer_rating"],"ngspass"),
        "rushing":("rush_attempts",["efficiency","percent_attempts_gte_eight_defenders","avg_time_to_los","rush_yards_over_expected_per_att","rush_pct_over_expected"],"ngsrush"),
        "receiving":("targets",["avg_cushion","avg_separation","avg_intended_air_yards","catch_percentage","avg_yac","avg_expected_yac","avg_yac_above_expectation"],"ngsrec"),
    }
    for kind,(weight,metrics,stem) in configs.items():
        d=load_ngs_years(years,kind)
        if d.empty or "team_abbr" not in d.columns: continue
        d=d[pd.to_numeric(d.get("week"),errors="coerce").fillna(0)>0].copy()
        d["team"]=d["team_abbr"].map(norm_team)
        d["season"]=pd.to_numeric(d["season"],errors="coerce"); d["week"]=pd.to_numeric(d["week"],errors="coerce")
        if kind=="receiving" and weight not in d.columns:
            weight="receptions" if "receptions" in d.columns else weight
        mets=[c for c in metrics if c in d.columns]
        rows=[]
        for k,g in d.groupby(["season","week","team"],dropna=False):
            r={"season":k[0],"week":k[1],"team":k[2]}
            for c in mets:
                r[f"ngs_{kind}_{c}"]=_weighted_mean(g,c,weight)
            rows.append(r)
        t=pd.DataFrame(rows)
        for side,team in [("home",home),("away",away)]:
            h=t[t.team==team].copy()
            h=h[(h.season<season)|((h.season==season)&(h.week<week))].sort_values(["season","week"])
            if h.empty: continue
            any_got=True
            for m in [c for c in t.columns if c.startswith(f"ngs_{kind}_")]:
                rv=_roll_values(h,m,season)
                for suf in ["r5","ewm","std"]:
                    out[f"{side}_{stem}_pre_{m}_{suf}"]=rv[suf]
    return out, any_got


def build_snap_features(season:int, week:int, home:str, away:str):
    d=load_snap_years(tuple(sorted(set([season-1,season]))))
    if d.empty or not {"team","season","week","game_id"}.issubset(d.columns):
        return {}, False
    s=d.copy(); s["team"]=s["team"].map(norm_team)
    s["season"]=pd.to_numeric(s["season"],errors="coerce"); s["week"]=pd.to_numeric(s["week"],errors="coerce")
    s["offense_pct"]=pd.to_numeric(s.get("offense_pct"),errors="coerce"); s["defense_pct"]=pd.to_numeric(s.get("defense_pct"),errors="coerce")
    if s["offense_pct"].dropna().median()>1.5: s["offense_pct"]/=100
    if s["defense_pct"].dropna().median()>1.5: s["defense_pct"]/=100
    rows=[]
    for k,g in s.groupby(["season","week","team","game_id"],dropna=False):
        idcol="pfr_player_id" if "pfr_player_id" in g.columns else ("player_id" if "player_id" in g.columns else None)
        os=set(g.loc[g.offense_pct.fillna(0)>=.5,idcol].dropna().astype(str)) if idcol else set()
        ds=set(g.loc[g.defense_pct.fillna(0)>=.5,idcol].dropna().astype(str)) if idcol else set()
        pos=g.get("position",pd.Series("",index=g.index)).astype(str).str.upper()
        ol=g[pos.isin(["T","OT","G","OG","C","OL"])]
        rows.append({"season":k[0],"week":k[1],"team":k[2],"game_id":k[3],
                     "snap_off_50pct_players":len(os),"snap_def_50pct_players":len(ds),
                     "snap_ol_total":float(pd.to_numeric(ol.get("offense_snaps"),errors="coerce").sum()) if "offense_snaps" in ol.columns else np.nan,
                     "_os":os,"_ds":ds})
    t=pd.DataFrame(rows).sort_values(["team","season","week","game_id"])
    lasto={};lastd={};oc=[];dc=[]
    for _,r in t.iterrows():
        team=r.team; po=lasto.get(team,set()); pdv=lastd.get(team,set()); co=r._os; cd=r._ds
        oc.append(len(co&po)/max(1,len(co|po)) if po else np.nan)
        dc.append(len(cd&pdv)/max(1,len(cd|pdv)) if pdv else np.nan)
        lasto[team]=co;lastd[team]=cd
    t["snap_off_continuity_jaccard"]=oc; t["snap_def_continuity_jaccard"]=dc
    out={}; got=False
    for side,team in [("home",home),("away",away)]:
        h=t[t.team==team].copy(); h=h[(h.season<season)|((h.season==season)&(h.week<week))].sort_values(["season","week","game_id"])
        if h.empty: continue
        got=True
        for m in ["snap_off_50pct_players","snap_def_50pct_players","snap_ol_total","snap_off_continuity_jaccard","snap_def_continuity_jaccard"]:
            rv=_roll_values(h,m,season)
            for suf in ["r3","r5","r8","ewm","std"]:
                out[f"{side}_snap_pre_{m}_{suf}"]=rv[suf]
    return out,got


def _pg(pos):
    return POSITION_GROUP.get(str(pos).upper().strip(),"OTHER")


def build_depth_features(season:int, game_date, home:str, away:str):
    d=load_depth_current(season)
    if d.empty or not {"team","pos_rank"}.issubset(d.columns):
        return {},False
    x=d.copy(); x["team"]=x["team"].map(norm_team); x["_rank"]=pd.to_numeric(x["pos_rank"],errors="coerce")
    poscol="pos_abb" if "pos_abb" in x.columns else ("position" if "position" in x.columns else "pos_name" if "pos_name" in x.columns else None)
    if not poscol: return {},False
    x["_pg"]=x[poscol].map(_pg)
    idcol="gsis_id" if "gsis_id" in x.columns else ("espn_id" if "espn_id" in x.columns else "player_name" if "player_name" in x.columns else None)
    if not idcol: return {},False
    if "dt" in x.columns:
        x["_dt"]=pd.to_datetime(x["dt"],errors="coerce",utc=True)
        cutoff=pd.Timestamp(game_date).tz_localize("UTC") if pd.Timestamp(game_date).tzinfo is None else pd.Timestamp(game_date).tz_convert("UTC")
        x=x[x["_dt"].notna() & (x["_dt"] < cutoff)]
    out={}; got=False
    for side,team in [("home",home),("away",away)]:
        z=x[x.team==team].copy()
        if z.empty: continue
        if "_dt" in z.columns:
            dates=sorted(z["_dt"].dropna().unique())
            if not dates: continue
            cur=z[z["_dt"]==dates[-1]]
            prev=z[z["_dt"]==dates[-2]] if len(dates)>1 else pd.DataFrame()
        else:
            cur=z;prev=pd.DataFrame()
        stt=cur[cur._rank==1]
        got=True
        ids=set(stt[idcol].dropna().astype(str))
        pids=set(prev.loc[prev._rank==1,idcol].dropna().astype(str)) if len(prev) else set()
        q=stt[stt._pg=="QB"][idcol].dropna().astype(str)
        pq=prev[prev._rank==1]; pq=pq[pq._pg=="QB"][idcol].dropna().astype(str) if len(prev) else pd.Series(dtype=str)
        out[f"{side}_depth_starter_count"]=len(ids)
        out[f"{side}_depth_qb1_count"]=int((stt._pg=="QB").sum())
        out[f"{side}_depth_ol1_count"]=int((stt._pg=="OL").sum())
        out[f"{side}_depth_skill1_count"]=int(stt._pg.isin(["RB","SKILL"]).sum())
        out[f"{side}_depth_dl1_count"]=int((stt._pg=="DL").sum())
        out[f"{side}_depth_lb1_count"]=int((stt._pg=="LB").sum())
        out[f"{side}_depth_db1_count"]=int((stt._pg=="DB").sum())
        out[f"{side}_depth_starter_continuity"]=len(ids&pids)/max(1,len(ids|pids)) if pids else np.nan
        out[f"{side}_depth_qb1_change"]=float(len(q)>0 and len(pq)>0 and q.iloc[0]!=pq.iloc[0]) if len(pq) else np.nan
    return out,got


def selected_feature_names(bundle):
    feats=bundle["features"]
    idx=set()
    for key in ["home_score_hgb","away_score_hgb","margin_hgb"]:
        try:
            sel=bundle[key].named_steps["sel"]
            idx.update(int(i) for i in sel.get_support(indices=True))
        except Exception:
            pass
    return [feats[i] for i in sorted(idx)] if idx else list(feats)


def build_side_prediction(
    *, root: Path, schedule: pd.DataFrame, game, prev_pbp: pd.DataFrame, cur_pbp: pd.DataFrame,
    home: str, away: str, season: int, week: int, market_total: float,
    over_odds: float, under_odds: float, home_spread: float, home_spread_odds: float,
    away_spread_odds: float, home_ml: float, away_ml: float, dome: float,
):
    bundle, meta=load_side_bundle(str(root))
    features=list(bundle["features"])
    live={c:np.nan for c in features}

    # Model uses positive spread_line when HOME is favored. Sportsbook UI uses -6 for home favorite.
    spread_line=-float(home_spread)
    live.update({
        "season":float(season),"week":float(week),
        "away_rest":float(_val(game,"away_rest",7.0)),"home_rest":float(_val(game,"home_rest",7.0)),
        "away_moneyline":float(away_ml),"home_moneyline":float(home_ml),
        "spread_line":spread_line,"away_spread_odds":float(away_spread_odds),"home_spread_odds":float(home_spread_odds),
        "total_line":float(market_total),"under_odds":float(under_odds),"over_odds":float(over_odds),
        "div_game":float(_val(game,"div_game",0.0)),
        "is_dome_or_closed":float(dome),"is_open_roof":float(1.0-dome if str(_val(game,"roof","")).lower()=="open" else 0.0),
    })
    mlh,mla,mlhold=devig(home_ml,away_ml); sph,spa,sphold=devig(home_spread_odds,away_spread_odds); ov,un,thold=devig(over_odds,under_odds)
    live.update({
        "market_ml_side1_devig":mlh,"market_ml_side2_devig":mla,"market_ml_hold":mlhold,
        "market_spread_side1_devig":sph,"market_spread_side2_devig":spa,"market_spread_hold":sphold,
        "market_total_side1_devig":ov,"market_total_side2_devig":un,"market_total_hold":thold,
        "market_home_points":(float(market_total)+spread_line)/2.0,
        "market_away_points":(float(market_total)-spread_line)/2.0,
        "rest_diff_home_minus_away":float(_val(game,"home_rest",7.0))-float(_val(game,"away_rest",7.0)),
    })

    # Team / matchup / possession features.
    fam={"team":False,"qb":False,"qbr":False,"ngs":False,"snap":False,"depth":False,"injuries":False}
    try:
        z,ok=build_team_features(prev_pbp,cur_pbp,home,away,season,week); live.update({k:v for k,v in z.items() if k in live}); fam["team"]=bool(ok.get("team"))
    except Exception:
        pass

    home_qb=_val(game,"home_qb_id",None); away_qb=_val(game,"away_qb_id",None)
    try:
        z,ok=build_qb_features(season,week,home_qb,away_qb); live.update({k:v for k,v in z.items() if k in live}); fam["qb"]=ok
    except Exception:
        pass
    try:
        z,ok=build_qbr_features(season,week,home,away); live.update({k:v for k,v in z.items() if k in live}); fam["qbr"]=ok
    except Exception:
        pass
    try:
        z,ok=build_ngs_features(season,week,home,away); live.update({k:v for k,v in z.items() if k in live}); fam["ngs"]=ok
    except Exception:
        pass
    try:
        z,ok=build_snap_features(season,week,home,away); live.update({k:v for k,v in z.items() if k in live}); fam["snap"]=ok
    except Exception:
        pass
    try:
        z,ok=build_depth_features(season,pd.Timestamp(_val(game,"gameday")),home,away); live.update({k:v for k,v in z.items() if k in live}); fam["depth"]=ok
    except Exception:
        pass

    # QB-change flags: reproduce the historical builder from prior schedule starters when possible.
    try:
        sched = schedule.copy()
        sched["gameday"] = pd.to_datetime(sched["gameday"], errors="coerce")
        cutoff = pd.Timestamp(_val(game, "gameday"))
        past = sched[sched["gameday"] < cutoff].copy()
        def previous_qb(team):
            rows=[]
            if {"home_team","home_qb_id"}.issubset(past.columns):
                h=past[past["home_team"].map(norm_team)==team][["gameday","home_qb_id"]].rename(columns={"home_qb_id":"qb"})
                rows.append(h)
            if {"away_team","away_qb_id"}.issubset(past.columns):
                a=past[past["away_team"].map(norm_team)==team][["gameday","away_qb_id"]].rename(columns={"away_qb_id":"qb"})
                rows.append(a)
            if not rows: return None
            z=pd.concat(rows,ignore_index=True).dropna(subset=["qb"]).sort_values("gameday")
            return str(z.iloc[-1]["qb"]) if len(z) else None
        hp=previous_qb(home); ap=previous_qb(away)
        live["home_qb_change"] = float(hp is not None and home_qb is not None and str(hp)!=str(home_qb)) if home_qb is not None else np.nan
        live["away_qb_change"] = float(ap is not None and away_qb is not None and str(ap)!=str(away_qb)) if away_qb is not None else np.nan
    except Exception:
        live["home_qb_change"] = live.get("home_depth_qb1_change", np.nan)
        live["away_qb_change"] = live.get("away_depth_qb1_change", np.nan)

    X=pd.DataFrame([[live.get(c,np.nan) for c in features]],columns=features)
    def pred_pair(ridge_key,hgb_key,prior):
        r=float(bundle[ridge_key].predict(X)[0]); h=float(bundle[hgb_key].predict(X)[0])
        return float(prior)+.25*r+.75*h
    mh=live["market_home_points"]; ma=live["market_away_points"]
    home_pts=pred_pair("home_score_ridge","home_score_hgb",mh)
    away_pts=pred_pair("away_score_ridge","away_score_hgb",ma)
    direct_margin=pred_pair("margin_ridge","margin_hgb",spread_line)
    score_margin=home_pts-away_pts
    final_margin=.25*score_margin+.75*direct_margin
    sig=float(bundle["sigma_margin"])
    p_home_win=0.5*(1+math.erf(final_margin/(sig*math.sqrt(2))))
    p_home_cover=0.5*(1+math.erf((final_margin-spread_line)/(sig*math.sqrt(2))))
    p_away_win=1-p_home_win; p_away_cover=1-p_home_cover

    selected=selected_feature_names(bundle)
    vals=pd.Series({c:live.get(c,np.nan) for c in selected})
    coverage=float(vals.notna().mean()) if len(vals) else 0.0
    missing=[c for c in selected if pd.isna(live.get(c,np.nan))]

    return {
        "model_name":meta.get("model_name","NFL V2.2 Side Model"),
        "status":meta.get("status","Frozen for 2026 forward testing"),
        "predicted_margin":final_margin,
        "score_component_home":home_pts,"score_component_away":away_pts,
        "direct_margin":direct_margin,"score_margin":score_margin,
        "p_home_win":p_home_win,"p_away_win":p_away_win,
        "p_home_cover":p_home_cover,"p_away_cover":p_away_cover,
        "fair_home_ml":fair_american(p_home_win),"fair_away_ml":fair_american(p_away_win),
        "fair_home_spread":fair_american(p_home_cover),"fair_away_spread":fair_american(p_away_cover),
        "home_ml_ev":ev(p_home_win,home_ml),"away_ml_ev":ev(p_away_win,away_ml),
        "home_spread_ev":ev(p_home_cover,home_spread_odds),"away_spread_ev":ev(p_away_cover,away_spread_odds),
        "coverage":coverage,"selected_feature_count":len(selected),"missing_selected":missing,
        "families":fam,"spread_line_model":spread_line,
        "home_qb_id":home_qb,"away_qb_id":away_qb,
    }
