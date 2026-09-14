from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from statistics import NormalDist
from typing import Any
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

from v2_side_live_v23 import (
    TEAM_FULL,
    build_side_prediction_v23,
    ev,
    fair_american,
    load_side_bundle,
    norm_team,
)

# -----------------------------------------------------------------------------
# V2.4 referee layer
# -----------------------------------------------------------------------------
# This layer is intentionally small. Development-only tests on 2021-2025
# walk-forward predictions found only a modest signal, so the referee overlay is
# capped at +/- 0.25 points. 2026 is the forward test.
V24_REF_POINT_CAP = 0.25
V24_REF_SIGNAL_SCALE = 1.50
V24_MIN_REF_GAMES = 6

FOOTBALL_ZEBRAS_SEARCH = "https://www.footballzebras.com/?s={query}"

# A dated cache makes Week 1 work even if Football Zebras blocks a cloud request.
# Later weeks are discovered from the site automatically; the nflverse schedule
# itself is also checked first because it sometimes fills assignments pregame.
WEEK1_2026_CACHE = {
    "2026_01_NE_SEA": "Adrian Hill",
    "2026_01_SF_LA": "Alex Kemp",
    "2026_01_CHI_CAR": "Brad Allen",
    "2026_01_TB_CIN": "John Hussey",
    "2026_01_NO_DET": "Alan Eck",
    "2026_01_BUF_HOU": "Alex Moore",
    "2026_01_BAL_IND": "Scott Novak",
    "2026_01_CLE_JAX": "Craig Wrolstad",
    "2026_01_ATL_PIT": "Brad Rogers",
    "2026_01_NYJ_TEN": "Land Clark",
    "2026_01_ARI_LAC": "Ron Torbert",
    "2026_01_MIA_LV": "Clete Blakeman",
    "2026_01_GB_MIN": "Shawn Hochuli",
    "2026_01_WAS_PHI": "Shawn Smith",
    "2026_01_DAL_NYG": "Carl Cheffers",
    "2026_01_DEN_KC": "Clay Martin",
}

TEAM_NICK = {k: v.split()[-1] for k, v in TEAM_FULL.items()}


def _clean_name(x: Any) -> str:
    if x is None:
        return ""
    s = unicodedata.normalize("NFKD", str(x))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _valid_ref_name(s: str) -> bool:
    s = _clean_name(s)
    if not s or len(s) > 60:
        return False
    bad = (
        "p.m", "a.m", "nfl", "espn", "abc", "nbc", "cbs", "fox", "peacock",
        "referee assignments", "week ", "published", "football zebras", "crew",
        "network", "amazon", "netflix", "prime", "off this week",
    )
    sl = s.lower()
    if any(x in sl for x in bad):
        return False
    # Most NFL referee names are 2-4 alphabetic name tokens.
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z.'-]+(?:\s+[A-Za-z][A-Za-z.'-]+){1,3}", s))


def _game_id(game: Any, season: int, week: int, away: str, home: str) -> str:
    try:
        gid = game.get("game_id")
        if gid is not None and not pd.isna(gid) and str(gid).strip():
            return str(gid)
    except Exception:
        pass
    return f"{int(season)}_{int(week):02d}_{away}_{home}"


def _ref_from_schedule_game(game: Any) -> str:
    try:
        x = game.get("referee")
    except Exception:
        x = None
    if x is not None and not pd.isna(x):
        s = _clean_name(x)
        if _valid_ref_name(s):
            return s
    return ""


def _aliases(team: str) -> list[str]:
    full = TEAM_FULL.get(team, team)
    nick = TEAM_NICK.get(team, team)
    vals = [full, nick]
    # City/short name helps with occasional headline formatting.
    if full and " " in full:
        vals.append(full.rsplit(" ", 1)[0])
    return list(dict.fromkeys(_clean_name(v) for v in vals if v))


def _extract_assignment_from_html(html: str, away: str, home: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    lines = [_clean_name(x) for x in soup.get_text("\n", strip=True).splitlines()]
    lines = [x for x in lines if x]
    away_alias = _aliases(away)
    home_alias = _aliases(home)

    def matchup_line(line: str) -> bool:
        ll = line.lower()
        return any(f"{a.lower()} at {h.lower()}" in ll for a in away_alias for h in home_alias)

    for i, line in enumerate(lines):
        if not matchup_line(line):
            continue
        for nxt in lines[i + 1 : i + 8]:
            if _valid_ref_name(nxt):
                return _clean_name(nxt)
    return ""


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_url(url: str) -> tuple[str, str]:
    try:
        r = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; NFL-Betting-Lab/1.0)"},
        )
        r.raise_for_status()
        return r.text, r.url
    except Exception:
        return "", url


def _football_zebras_candidate_urls(season: int, week: int, gameday: Any) -> list[str]:
    urls: list[str] = []
    try:
        dt = pd.Timestamp(gameday)
        months = [int(dt.month), int((dt - pd.Timedelta(days=10)).month), int((dt + pd.Timedelta(days=10)).month)]
    except Exception:
        months = [9, 10, 11, 12, 1]
    for month in list(dict.fromkeys(months + [9, 10, 11, 12, 1])):
        year_for_path = season if month != 1 else season + 1
        urls.append(
            f"https://www.footballzebras.com/{year_for_path}/{month:02d}/week-{int(week)}-referee-assignments-{int(season)}/"
        )
    return urls


@st.cache_data(ttl=1800, show_spinner=False)
def _football_zebras_assignment(season: int, week: int, gameday_iso: str, away: str, home: str) -> dict[str, Any]:
    # First try predictable article URLs.
    for url in _football_zebras_candidate_urls(season, week, gameday_iso):
        html, final_url = _fetch_url(url)
        if not html:
            continue
        ref = _extract_assignment_from_html(html, away, home)
        if ref:
            return {"referee": ref, "source": "Football Zebras", "url": final_url}

    # Fallback: site search, then follow likely Week N assignment links.
    query = quote_plus(f"week {int(week)} referee assignments {int(season)}")
    html, search_url = _fetch_url(FOOTBALL_ZEBRAS_SEARCH.format(query=query))
    if html:
        soup = BeautifulSoup(html, "html.parser")
        hrefs = []
        for a in soup.find_all("a", href=True):
            href = str(a.get("href", ""))
            text = _clean_name(a.get_text(" ", strip=True)).lower()
            key = f"week-{int(week)}-referee-assignments"
            if key in href.lower() or (f"week {int(week)}" in text and "referee assignments" in text):
                if href.startswith("/"):
                    href = "https://www.footballzebras.com" + href
                if href.startswith("http"):
                    hrefs.append(href)
        for href in list(dict.fromkeys(hrefs))[:6]:
            page, final_url = _fetch_url(href)
            ref = _extract_assignment_from_html(page, away, home) if page else ""
            if ref:
                return {"referee": ref, "source": "Football Zebras", "url": final_url}

    return {"referee": "", "source": "unavailable", "url": search_url}


def resolve_head_referee(
    *, schedule: pd.DataFrame, game: Any, season: int, week: int, away: str, home: str,
) -> dict[str, Any]:
    # 1) nflverse schedule field if already populated.
    ref = _ref_from_schedule_game(game)
    if ref:
        return {"referee": ref, "source": "nflverse schedule", "url": ""}

    gid = _game_id(game, season, week, away, home)

    # 2) Dated built-in cache for the current launch week.
    if gid in WEEK1_2026_CACHE:
        return {
            "referee": WEEK1_2026_CACHE[gid],
            "source": "Football Zebras Week 1 cache",
            "url": "https://www.footballzebras.com/2026/09/week-1-referee-assignments-2026/",
        }

    # 3) Live free assignment scrape.
    gameday = None
    try:
        gameday = game.get("gameday")
    except Exception:
        pass
    return _football_zebras_assignment(int(season), int(week), str(gameday or ""), away, home)


def _prep_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    s = schedule.copy()
    if "game_id" not in s.columns:
        return pd.DataFrame()
    s["gameday"] = pd.to_datetime(s.get("gameday"), errors="coerce")
    s["season"] = pd.to_numeric(s.get("season"), errors="coerce")
    s["week"] = pd.to_numeric(s.get("week"), errors="coerce")
    s["spread_line"] = pd.to_numeric(s.get("spread_line"), errors="coerce")
    s["total_line"] = pd.to_numeric(s.get("total_line"), errors="coerce")
    s["home_score"] = pd.to_numeric(s.get("home_score"), errors="coerce")
    s["away_score"] = pd.to_numeric(s.get("away_score"), errors="coerce")
    s["home_margin"] = s["home_score"] - s["away_score"]
    s["home_ats_margin"] = s["home_margin"] - s["spread_line"]
    s["total_points"] = s["home_score"] + s["away_score"]
    s["total_margin_vs_line"] = s["total_points"] - s["total_line"]
    s["referee_clean"] = s.get("referee", pd.Series("", index=s.index)).map(_clean_name)

    # nflverse can lag current-season referee fields even after games are final.
    # Hydrate completed current-season rows from our dated cache or Football
    # Zebras so referee current-season stats actually advance week by week.
    need_ref = (
        s["home_margin"].notna()
        & s["referee_clean"].eq("")
        & s["season"].notna()
        & (s["season"] >= 2026)
    )
    for idx, row in s.loc[need_ref].iterrows():
        gid = str(row.get("game_id") or "")
        ref = _clean_name(WEEK1_2026_CACHE.get(gid, ""))
        if not ref:
            try:
                away = norm_team(row.get("away_team"))
                home = norm_team(row.get("home_team"))
                info = _football_zebras_assignment(
                    int(row.get("season")), int(row.get("week")), str(row.get("gameday") or ""), away, home
                )
                ref = _clean_name(info.get("referee", ""))
            except Exception:
                ref = ""
        if _valid_ref_name(ref):
            s.at[idx, "referee_clean"] = ref

    s["role"] = np.where(s["spread_line"] > 0, "HOME_FAV", np.where(s["spread_line"] < 0, "HOME_DOG", "PICK"))
    return s


def _ats_record(df: pd.DataFrame, side: str = "home") -> dict[str, int]:
    if df.empty or "home_ats_margin" not in df.columns:
        return {"wins": 0, "losses": 0, "pushes": 0, "games": 0}
    x = pd.to_numeric(df["home_ats_margin"], errors="coerce").dropna()
    if side == "away":
        x = -x
    return {
        "wins": int((x > 0).sum()),
        "losses": int((x < 0).sum()),
        "pushes": int((x == 0).sum()),
        "games": int(len(x)),
    }


def _ou_record(df: pd.DataFrame) -> dict[str, int]:
    if df.empty or "total_margin_vs_line" not in df.columns:
        return {"over": 0, "under": 0, "pushes": 0, "games": 0}
    x = pd.to_numeric(df["total_margin_vs_line"], errors="coerce").dropna()
    return {
        "over": int((x > 0).sum()),
        "under": int((x < 0).sum()),
        "pushes": int((x == 0).sum()),
        "games": int(len(x)),
    }


def _mean_shrunk(df: pd.DataFrame, prior_games: float) -> tuple[float, float, int]:
    if df.empty:
        return 0.0, float("nan"), 0
    x = pd.to_numeric(df["home_ats_margin"], errors="coerce").dropna()
    n = int(len(x))
    if not n:
        return 0.0, float("nan"), 0
    raw = float(x.mean())
    shr = float(raw * n / (n + float(prior_games)))
    return shr, raw, n


def compute_referee_tracker(
    *, schedule: pd.DataFrame, referee: str, game: Any, season: int, week: int,
    away: str, home: str, home_spread: float,
) -> dict[str, Any]:
    ref = _clean_name(referee)
    if not ref:
        return {"available": False, "referee": "", "reason": "No head referee assignment found."}

    s = _prep_schedule(schedule)
    if s.empty:
        return {"available": False, "referee": ref, "reason": "Schedule history unavailable."}

    try:
        cutoff = pd.Timestamp(game.get("gameday"))
    except Exception:
        cutoff = pd.Timestamp.utcnow().tz_localize(None)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)

    # Strictly pregame: only games completed before selected kickoff date.
    past = s[(s["gameday"] < cutoff) & s["home_margin"].notna() & s["spread_line"].notna()].copy()
    if "game_type" in past.columns:
        # Include regular season and playoffs; exclude preseason if a provider ever adds it.
        past = past[~past["game_type"].astype(str).str.upper().isin(["PRE", "P"])]
    rp = past[past["referee_clean"].str.casefold() == ref.casefold()].sort_values(["gameday", "game_id"])
    if rp.empty:
        return {"available": False, "referee": ref, "reason": "No prior games found for this referee."}

    # Sportsbook UI: negative home spread means home favorite.
    model_spread_line = -float(home_spread)
    role = "HOME_FAV" if model_spread_line > 0 else ("HOME_DOG" if model_spread_line < 0 else "PICK")
    role_df = rp[rp["role"] == role]
    recent3 = rp[rp["season"] >= int(season) - 3]
    current = rp[rp["season"] == int(season)]
    last16 = rp.tail(16)

    c_shr, c_raw, c_n = _mean_shrunk(rp, 40)
    r_shr, r_raw, r_n = _mean_shrunk(role_df, 30)
    s_shr, s_raw, s_n = _mean_shrunk(current, 12)
    l_shr, l_raw, l_n = _mean_shrunk(last16, 20)

    signal = 0.20 * c_shr + 0.40 * r_shr + 0.15 * s_shr + 0.25 * l_shr
    if c_n < V24_MIN_REF_GAMES:
        adjustment = 0.0
    else:
        adjustment = float(V24_REF_POINT_CAP * math.tanh(signal / V24_REF_SIGNAL_SCALE))

    if role == "HOME_FAV":
        role_label = "Road underdogs"
        role_record = _ats_record(role_df, side="away")
    elif role == "HOME_DOG":
        role_label = "Home underdogs"
        role_record = _ats_record(role_df, side="home")
    else:
        role_label = "Home teams in pick'em games"
        role_record = _ats_record(role_df, side="home")

    return {
        "available": True,
        "referee": ref,
        "career_games": int(len(rp)),
        "recent3_games": int(len(recent3)),
        "current_season_games": int(len(current)),
        "last16_games": int(len(last16)),
        "career_home_ats": _ats_record(rp, side="home"),
        "recent3_home_ats": _ats_record(recent3, side="home"),
        "last16_home_ats": _ats_record(last16, side="home"),
        "role": role,
        "role_label": role_label,
        "role_ats": role_record,
        "career_ou": _ou_record(rp),
        "recent3_ou": _ou_record(recent3),
        "career_mean_ats_residual": c_raw,
        "role_mean_ats_residual": r_raw,
        "last16_mean_ats_residual": l_raw,
        "current_season_mean_ats_residual": s_raw,
        "shrunk_components": {
            "career": c_shr,
            "same_market_role": r_shr,
            "current_season": s_shr,
            "last16": l_shr,
        },
        "signal_points": float(signal),
        "adjustment_points": float(adjustment),
        "point_cap": float(V24_REF_POINT_CAP),
        "signal_scale": float(V24_REF_SIGNAL_SCALE),
        "cutoff": str(cutoff.date()),
    }


def _shift_probability(p: float, z_shift: float) -> float:
    nd = NormalDist()
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return float(nd.cdf(nd.inv_cdf(p) + float(z_shift)))


def build_side_prediction_v24(*, referee_override: str | None = None, **kwargs):
    """
    V2.4 challenger = frozen V2.3 + a tiny, leakage-safe head-referee overlay.

    Referee history is computed only from games before the selected game date.
    The overlay uses shrunk ATS residuals (career, same market role, current
    season, last 16) and is capped at +/-0.25 points. The current assignment is
    sourced from nflverse/Football Zebras or a manual override.
    """
    base = build_side_prediction_v23(**kwargs)

    schedule = kwargs["schedule"]
    game = kwargs["game"]
    season = int(kwargs["season"])
    week = int(kwargs["week"])
    away = str(kwargs["away"])
    home = str(kwargs["home"])

    auto = resolve_head_referee(
        schedule=schedule, game=game, season=season, week=week, away=away, home=home
    )
    chosen = _clean_name(referee_override) if referee_override else _clean_name(auto.get("referee", ""))
    assignment = dict(auto)
    if referee_override and chosen:
        assignment = {"referee": chosen, "source": "manual override", "url": auto.get("url", ""), "auto_referee": auto.get("referee", "")}

    tracker = compute_referee_tracker(
        schedule=schedule,
        referee=chosen,
        game=game,
        season=season,
        week=week,
        away=away,
        home=home,
        home_spread=float(kwargs["home_spread"]),
    )
    ref_adj = float(tracker.get("adjustment_points", 0.0)) if tracker.get("available") else 0.0

    # Margin: keep V2.3 intact, then allow only the small referee overlay.
    base_margin = float(base["predicted_margin"])
    new_margin = base_margin + ref_adj

    # Probability influence is intentionally tiny: +/-0.25 points over sigma ~=
    # 12-13 points is only a small z-score movement.
    try:
        bundle, _ = load_side_bundle(str(Path(kwargs["root"])))
        sigma = max(6.0, float(bundle.get("sigma_margin", 12.75)))
    except Exception:
        sigma = 12.75
    z_shift = ref_adj / sigma

    p_home_cover = _shift_probability(float(base["p_home_cover"]), z_shift)
    p_home_win = _shift_probability(float(base["p_home_win"]), z_shift)
    p_away_cover = 1.0 - p_home_cover
    p_away_win = 1.0 - p_home_win

    home_ml = float(kwargs["home_ml"])
    away_ml = float(kwargs["away_ml"])
    home_spread_odds = float(kwargs["home_spread_odds"])
    away_spread_odds = float(kwargs["away_spread_odds"])

    out = dict(base)
    out.update({
        "model_name": "NFL V2.4 Referee Challenger",
        "status": "2026 referee-layer forward test",
        "v23_predicted_margin": base_margin,
        "predicted_margin": new_margin,
        "p_home_win": p_home_win,
        "p_away_win": p_away_win,
        "p_home_cover": p_home_cover,
        "p_away_cover": p_away_cover,
        "fair_home_ml": fair_american(p_home_win),
        "fair_away_ml": fair_american(p_away_win),
        "fair_home_spread": fair_american(p_home_cover),
        "fair_away_spread": fair_american(p_away_cover),
        "home_ml_ev": ev(p_home_win, home_ml),
        "away_ml_ev": ev(p_away_win, away_ml),
        "home_spread_ev": ev(p_home_cover, home_spread_odds),
        "away_spread_ev": ev(p_away_cover, away_spread_odds),
        "referee_assignment": assignment,
        "referee_tracker": tracker,
        "referee_adjustment_points": ref_adj,
        "referee_probability_z_shift": z_shift,
    })
    return out
