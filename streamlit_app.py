
from __future__ import annotations

import io
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

from market_scanner import (
    CORE_PROP_MARKETS, PROP_MARKET_LABELS, combine_prop_payloads,
    core_arbitrage_table, core_best_lines, core_middle_table,
    prop_arbitrage_table, prop_value_table,
    SPORT_KEYS, FUTURE_SPORT_KEYS, SPORT_PROP_MARKETS, SPORT_CORE_PROP_KEYS,
    SPORT_ALTERNATE_KEYS, EXTRA_GAME_MARKETS, normalize_market_payloads,
    combine_normalized_offer_frames, build_opportunity_board, apply_watchlist,
    book_coverage_summary,
)

try:
    from v2_side_live import build_side_prediction
    V22_SIDE_IMPORT_ERROR = None
except Exception as _v22_exc:
    build_side_prediction = None
    V22_SIDE_IMPORT_ERROR = str(_v22_exc)

try:
    from v2_side_live_v23 import build_side_prediction_v23, get_free_breaking_news
    V23_SIDE_IMPORT_ERROR = None
except Exception as _v23_exc:
    build_side_prediction_v23 = None
    get_free_breaking_news = None
    V23_SIDE_IMPORT_ERROR = str(_v23_exc)

try:
    from v2_side_live_v24 import build_side_prediction_v24, resolve_head_referee
    V24_SIDE_IMPORT_ERROR = None
except Exception as _v24_exc:
    build_side_prediction_v24 = None
    resolve_head_referee = None
    V24_SIDE_IMPORT_ERROR = str(_v24_exc)

try:
    from v2_side_live_v25 import build_side_prediction_v25
    V25_SIDE_IMPORT_ERROR = None
except Exception as _v25_exc:
    build_side_prediction_v25 = None
    V25_SIDE_IMPORT_ERROR = str(_v25_exc)

# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="Betting Lab + Market Radar",
    page_icon="🏈",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
FORWARD_TEST_FILE = ROOT / "v2_2026_forward_test_snapshot.csv"
SCHEDULE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
ODDS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
ODDS_EVENTS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"
ODDS_EVENT_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

BOOKMAKER_OPTIONS = {
    "DraftKings": "draftkings",
    "FanDuel": "fanduel",
    "BetMGM": "betmgm",
    "BetRivers": "betrivers",
    "Novig": "novig",
    "Kalshi": "kalshi",
    "Polymarket": "polymarket",
    "ProphetX": "prophetx",
    "Caesars (paid Odds API feed)": "williamhill_us",
}
DEFAULT_BOOKS = [
    "DraftKings", "FanDuel", "BetMGM", "BetRivers",
    "Novig", "Kalshi", "Polymarket", "ProphetX",
]

# nflverse uses LA for the Rams.
ODDS_NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}
ABBR_TO_ODDS_NAME = {v: k for k, v in ODDS_NAME_TO_ABBR.items()}

# Stadium-area coordinates used directly for weather.
# This avoids city-name geocoding failures and ties the forecast to the venue area.
WEATHER_COORDS = {
    "ARI": (33.5276, -112.2626, "State Farm Stadium"),
    "ATL": (33.7554, -84.4008, "Mercedes-Benz Stadium"),
    "BAL": (39.2780, -76.6227, "M&T Bank Stadium"),
    "BUF": (42.7738, -78.7868, "Highmark Stadium"),
    "CAR": (35.2258, -80.8528, "Bank of America Stadium"),
    "CHI": (41.8623, -87.6167, "Soldier Field"),
    "CIN": (39.0955, -84.5161, "Paycor Stadium"),
    "CLE": (41.5061, -81.6995, "Huntington Bank Field"),
    "DAL": (32.7473, -97.0945, "AT&T Stadium"),
    "DEN": (39.7439, -105.0201, "Empower Field at Mile High"),
    "DET": (42.3400, -83.0456, "Ford Field"),
    "GB": (44.5013, -88.0622, "Lambeau Field"),
    "HOU": (29.6847, -95.4107, "NRG Stadium"),
    "IND": (39.7601, -86.1639, "Lucas Oil Stadium"),
    "JAX": (30.3239, -81.6373, "EverBank Stadium"),
    "KC": (39.0489, -94.4839, "Arrowhead Stadium"),
    "LV": (36.0908, -115.1830, "Allegiant Stadium"),
    "LAC": (33.9535, -118.3392, "SoFi Stadium"),
    "LA": (33.9535, -118.3392, "SoFi Stadium"),
    "MIA": (25.9580, -80.2389, "Hard Rock Stadium"),
    "MIN": (44.9738, -93.2581, "U.S. Bank Stadium"),
    "NE": (42.0909, -71.2643, "Gillette Stadium"),
    "NO": (29.9511, -90.0812, "Caesars Superdome"),
    "NYG": (40.8135, -74.0745, "MetLife Stadium"),
    "NYJ": (40.8135, -74.0745, "MetLife Stadium"),
    "PHI": (39.9008, -75.1675, "Lincoln Financial Field"),
    "PIT": (40.4468, -80.0158, "Acrisure Stadium"),
    "SEA": (47.5952, -122.3316, "Lumen Field"),
    "SF": (37.4030, -121.9700, "Levi's Stadium"),
    "TB": (27.9759, -82.5033, "Raymond James Stadium"),
    "TEN": (36.1665, -86.7713, "Nissan Stadium"),
    "WAS": (38.9076, -76.8645, "Northwest Stadium"),
}

# Roof classification. Retractable-roof games still need a game-day open/closed choice.
FIXED_ROOF = {"DET", "MIN", "NO", "LV", "LA", "LAC"}
RETRACTABLE_ROOF = {"ARI", "ATL", "DAL", "HOU", "IND"}

FEATURES = [
    "EPA_Matchup",
    "Pass_EPA_Matchup",
    "Rush_EPA_Matchup",
    "Success_Matchup",
    "Explosive_Matchup",
    "Plays_Matchup",
    "Turnover_Env",
    "Sack_Env",
    "Wind",
    "Cold_Units",
    "Dome",
    "Short_Rest",
    "Div_Game",
]

PBP_REQUIRED = [
    "game_id", "season", "season_type", "week", "posteam", "defteam",
    "pass", "rush", "epa", "yards_gained", "interception", "fumble_lost", "sack",
]

# V1 only needs PBP_REQUIRED. The extra fields let the frozen V2.2 side model
# reproduce its drive, matchup, pace and scoring features live.
PBP_COLS = sorted(set(PBP_REQUIRED + [
    "home_team", "away_team", "drive", "fixed_drive", "no_play", "down",
    "ydstogo", "yardline_100", "air_yards", "yards_after_catch", "success",
    "wp", "score_differential", "qb_dropback", "pass_attempt", "rush_attempt",
    "qb_scramble", "qb_kneel", "qb_spike", "touchdown", "td_team",
    "field_goal_result", "extra_point_result", "two_point_conv_result",
    "first_down", "shotgun", "no_huddle", "xpass", "cpoe",
    "third_down_converted", "fourth_down_converted",
]))

# -----------------------------
# Local model files
# -----------------------------
@st.cache_data
def load_model_files():
    coef_df = pd.read_csv(ROOT / "model_coefficients.csv")
    coefficients = dict(zip(coef_df["Feature"], coef_df["Coefficient"].astype(float)))
    holdout = pd.read_csv(ROOT / "backtest_summary.csv").iloc[0].to_dict()
    historical = pd.read_csv(ROOT / "historical_features.csv")
    return coefficients, holdout, historical

COEF, HOLDOUT, HISTORICAL = load_model_files()

@st.cache_data(show_spinner=False)
def load_forward_test_snapshot() -> pd.DataFrame:
    if not FORWARD_TEST_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(FORWARD_TEST_FILE)
    for c in ["model_home_margin", "home_spread", "p_home_win", "p_home_cover", "verified_away_score", "verified_home_score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def grade_forward_test(snapshot: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    if snapshot.empty:
        return pd.DataFrame()
    out = snapshot.copy()
    live_cols = [c for c in ["game_id", "away_score", "home_score"] if c in schedule.columns]
    if len(live_cols) == 3:
        live = schedule[live_cols].drop_duplicates("game_id").copy()
        live["away_score"] = pd.to_numeric(live["away_score"], errors="coerce")
        live["home_score"] = pd.to_numeric(live["home_score"], errors="coerce")
        out = out.merge(live, on="game_id", how="left", suffixes=("", "_live"))
    else:
        out["away_score"] = pd.NA
        out["home_score"] = pd.NA

    # Prefer live schedule results; verified Week 1 scores are a fallback in case
    # the public schedule feed lags. Pending games remain pending.
    out["final_away_score"] = pd.to_numeric(out.get("away_score"), errors="coerce")
    out["final_home_score"] = pd.to_numeric(out.get("home_score"), errors="coerce")
    if "verified_away_score" in out:
        out["final_away_score"] = out["final_away_score"].fillna(pd.to_numeric(out["verified_away_score"], errors="coerce"))
    if "verified_home_score" in out:
        out["final_home_score"] = out["final_home_score"].fillna(pd.to_numeric(out["verified_home_score"], errors="coerce"))

    out["completed"] = out["final_home_score"].notna() & out["final_away_score"].notna()
    out["actual_home_margin"] = out["final_home_score"] - out["final_away_score"]
    out["actual_home_ats_margin"] = out["actual_home_margin"] + out["home_spread"]
    out["actual_home_cover"] = out["actual_home_ats_margin"] > 0
    out["ats_push"] = out["actual_home_ats_margin"] == 0
    out["model_home_ats_pick"] = out["p_home_cover"] > 0.5
    out["ats_correct"] = out["completed"] & (~out["ats_push"]) & (out["model_home_ats_pick"] == out["actual_home_cover"])

    out["actual_home_win"] = out["actual_home_margin"] > 0
    out["model_home_su_pick"] = out["p_home_win"] > 0.5
    out["su_correct"] = out["completed"] & (out["model_home_su_pick"] == out["actual_home_win"])
    out["model_margin_abs_error"] = (out["model_home_margin"] - out["actual_home_margin"]).abs()
    out["market_home_margin"] = -out["home_spread"]
    out["market_margin_abs_error"] = (out["market_home_margin"] - out["actual_home_margin"]).abs()
    out["brier"] = (out["p_home_win"] - out["actual_home_win"].astype(float)) ** 2

    def _logloss(row):
        if not bool(row.get("completed")):
            return float("nan")
        p = min(max(float(row["p_home_win"]), 1e-9), 1 - 1e-9)
        y = 1.0 if bool(row["actual_home_win"]) else 0.0
        return -(y * math.log(p) + (1 - y) * math.log(1 - p))
    out["log_loss"] = out.apply(_logloss, axis=1)
    return out


def render_forward_test_panel(schedule: pd.DataFrame) -> None:
    snap = load_forward_test_snapshot()
    if snap.empty:
        return
    graded = grade_forward_test(snap, schedule)
    done = graded[graded["completed"]].copy()
    if done.empty:
        return

    nonpush = done[~done["ats_push"]]
    ats_w = int(nonpush["ats_correct"].sum())
    ats_l = int(len(nonpush) - ats_w)
    ats_p = int(done["ats_push"].sum())
    su_w = int(done["su_correct"].sum())
    su_l = int(len(done) - su_w)

    st.subheader("2026 Forward-Test Results")
    st.caption(
        "These are frozen pregame V2.2 predictions saved from the September 12 screenshots. "
        "The Wednesday NE–SEA and Thursday SF–LA games are excluded because the forward-test app was deployed after they were played. "
        "V2.3 and V2.4 were not persistently logged for every Week 1 game, so they are not backfilled after the fact."
    )
    a, b, c, d = st.columns(4)
    a.metric("Completed test games", f"{len(done)}/{len(graded)}")
    a2 = f"{ats_w}-{ats_l}" + (f"-{ats_p}" if ats_p else "")
    b.metric("V2.2 ATS direction", a2, f"{(ats_w / max(1, len(nonpush))):.1%}")
    c.metric("Straight-up direction", f"{su_w}-{su_l}", f"{(su_w / max(1, len(done))):.1%}")
    d.metric("V2.2 margin MAE", f"{done['model_margin_abs_error'].mean():.2f} pts")

    a, b, c, d = st.columns(4)
    a.metric("Market-line margin MAE", f"{done['market_margin_abs_error'].mean():.2f} pts")
    b.metric("V2.2 ML Brier", f"{done['brier'].mean():.4f}")
    c.metric("V2.2 ML log loss", f"{done['log_loss'].mean():.4f}")
    pending = graded[~graded["completed"]]
    d.metric("Pending", ", ".join((pending["away"] + " @ " + pending["home"]).tolist()) if len(pending) else "None")

    st.info(
        "Do not retrain the estimator weights after one week. Week 1 game data should become Week 2 inputs, while V2.2/V2.3/V2.4 stay frozen so the 2026 forward test remains clean."
    )

    with st.expander("Week 1 game-by-game forward-test grading"):
        t = graded.copy()
        t["Matchup"] = t["away"] + " @ " + t["home"]
        t["Final"] = t.apply(lambda r: f"{int(r['final_away_score'])}-{int(r['final_home_score'])}" if r["completed"] else "Pending", axis=1)
        t["ATS pick"] = t.apply(lambda r: (r["home"] if r["model_home_ats_pick"] else r["away"]) + " ATS", axis=1)
        t["ATS result"] = t.apply(lambda r: "Pending" if not r["completed"] else ("Push" if r["ats_push"] else ("W" if r["ats_correct"] else "L")), axis=1)
        t["SU pick"] = t.apply(lambda r: r["home"] if r["model_home_su_pick"] else r["away"], axis=1)
        t["SU result"] = t.apply(lambda r: "Pending" if not r["completed"] else ("W" if r["su_correct"] else "L"), axis=1)
        show = t[["Matchup", "Final", "model_home_margin", "home_spread", "ATS pick", "ATS result", "SU pick", "SU result"]].rename(
            columns={"model_home_margin": "V2.2 home margin", "home_spread": "Pregame home spread"}
        )
        st.dataframe(show, width="stretch", hide_index=True)
        st.caption("ATS direction grades the side whose model cover probability was above 50%. This is not the same thing as ROI on every positive-EV price.")

# -----------------------------
# Network data loaders
# -----------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def load_schedule() -> pd.DataFrame:
    r = requests.get(SCHEDULE_URL, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_pbp(season: int) -> pd.DataFrame:
    url = PBP_URL.format(season=season)
    # Reading from bytes gives clearer errors than handing pandas a redirecting URL.
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(
        io.BytesIO(r.content),
        compression="gzip",
        usecols=lambda c: c in PBP_COLS,
        low_memory=False,
    )
    missing = [c for c in PBP_REQUIRED if c not in df.columns]
    if missing:
        raise RuntimeError(f"nflverse play-by-play is missing expected columns: {missing}")
    return df

def _odds_api_usage_from_response(response) -> dict:
    """Extract quota counters returned by The Odds API."""
    def _as_int(name):
        value = response.headers.get(name)
        if value is None:
            return None
        try:
            return int(float(value))
        except Exception:
            return None

    remaining = _as_int("x-requests-remaining")
    used = _as_int("x-requests-used")
    last = _as_int("x-requests-last")
    limit = (remaining + used) if remaining is not None and used is not None else None
    return {
        "remaining": remaining,
        "used": used,
        "last": last,
        "limit": limit,
        "observed_at": datetime.now(timezone.utc).timestamp(),
    }


def remember_odds_api_usage(usage: dict | None):
    """Keep the freshest quota observation in session state.

    Cached API responses carry the timestamp from the original HTTP request, so an
    old cached response cannot overwrite a newer quota reading from a prop scan.
    """
    if not usage or usage.get("remaining") is None:
        return
    current = st.session_state.get("odds_api_usage") or {}
    if float(usage.get("observed_at") or 0) >= float(current.get("observed_at") or 0):
        st.session_state["odds_api_usage"] = usage
        if usage.get("last") is not None and int(usage.get("last") or 0) > 0:
            st.session_state["odds_api_last_charged"] = int(usage["last"])


def current_odds_api_usage() -> dict:
    return dict(st.session_state.get("odds_api_usage") or {})


def render_odds_api_credit_panel():
    usage = current_odds_api_usage()
    st.markdown("#### Odds API credits")
    if not api_key:
        st.caption("Add your Odds API key to see live quota usage.")
        return
    if usage.get("remaining") is None:
        st.caption("Quota will appear after the first Odds API request.")
        return

    remaining = int(usage["remaining"])
    used = int(usage.get("used") or 0)
    limit = usage.get("limit")
    last = usage.get("last")
    last_charged = st.session_state.get("odds_api_last_charged")
    if limit is not None and int(limit) > 0:
        st.progress(min(1.0, max(0.0, remaining / int(limit))))
        st.caption(f"**{remaining:,} remaining** · {used:,} used · {int(limit):,} quota")
    else:
        st.caption(f"**{remaining:,} remaining** · {used:,} used")
    if last_charged is not None:
        st.caption(f"Last charged request: **{int(last_charged):,} credit{'s' if int(last_charged) != 1 else ''}**")
    elif last is not None:
        st.caption(f"Last API request cost: **{int(last):,} credit{'s' if int(last) != 1 else ''}**")
    if remaining <= 25:
        st.error("Very low API quota — avoid broad player-prop scans.")
    elif remaining <= 75:
        st.warning("API quota is getting low. Prefer selected-game or fewer-market scans.")


@st.cache_data(ttl=120, show_spinner=False)
def fetch_odds(_api_key: str, _bookmakers_csv: str = ""):
    if not _api_key:
        return []
    params = {
        "apiKey": _api_key,
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
    }
    if _bookmakers_csv:
        params["bookmakers"] = _bookmakers_csv
    else:
        params["regions"] = "us"
    r = requests.get(ODDS_URL, params=params, timeout=30)
    if r.status_code == 401:
        raise RuntimeError("The Odds API rejected the key. Check the key and try again.")
    if r.status_code == 429:
        raise RuntimeError("The Odds API usage limit has been reached.")
    r.raise_for_status()
    return r.json(), _odds_api_usage_from_response(r)

@st.cache_data(ttl=120, show_spinner=False)
def fetch_nfl_events(_api_key: str):
    if not _api_key:
        return []
    r = requests.get(ODDS_EVENTS_URL, params={"apiKey": _api_key, "dateFormat": "iso"}, timeout=30)
    if r.status_code == 401:
        raise RuntimeError("The Odds API rejected the key.")
    if r.status_code == 429:
        raise RuntimeError("The Odds API usage limit has been reached.")
    r.raise_for_status()
    return r.json(), _odds_api_usage_from_response(r)

@st.cache_data(ttl=90, show_spinner=False)
def fetch_event_props(_api_key: str, event_id: str, markets_csv: str, bookmakers_csv: str):
    if not _api_key or not event_id or not markets_csv:
        return {}
    params = {
        "apiKey": _api_key,
        "markets": markets_csv,
        "oddsFormat": "american",
    }
    if bookmakers_csv:
        params["bookmakers"] = bookmakers_csv
    else:
        params["regions"] = "us"
    url = ODDS_EVENT_URL.format(event_id=event_id)
    r = requests.get(url, params=params, timeout=40)
    if r.status_code == 401:
        raise RuntimeError("The Odds API rejected the key.")
    if r.status_code == 404:
        return {}, _odds_api_usage_from_response(r)
    if r.status_code == 422:
        raise RuntimeError("The Odds API rejected one of the requested prop markets/bookmakers. Try fewer markets or books.")
    if r.status_code == 429:
        raise RuntimeError("The Odds API usage limit has been reached.")
    r.raise_for_status()
    return r.json(), _odds_api_usage_from_response(r)

# Generic multi-sport Odds API helpers used by Market Radar.
ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"


def _raise_market_api_error(r, context: str = "Odds API request"):
    if r.status_code == 401:
        raise RuntimeError("The Odds API rejected the key.")
    if r.status_code == 422:
        detail = ""
        try:
            detail = str((r.json() or {}).get("message") or "")
        except Exception:
            detail = r.text[:240] if getattr(r, "text", "") else ""
        if "williamhill_us" in str(getattr(r.request, "url", "")):
            detail = (detail + " Caesars is a paid-only Odds API bookmaker feed; turn Caesars off if you are on the free plan.").strip()
        raise RuntimeError(f"{context} was rejected (422). {detail}".strip())
    if r.status_code == 429:
        raise RuntimeError("The Odds API usage limit has been reached.")
    r.raise_for_status()


@st.cache_data(ttl=120, show_spinner=False)
def fetch_sport_events(_api_key: str, sport_key: str):
    if not _api_key:
        return [], {}
    url = f"{ODDS_API_BASE}/{sport_key}/events"
    r = requests.get(url, params={"apiKey": _api_key, "dateFormat": "iso"}, timeout=30)
    _raise_market_api_error(r, f"{sport_key} events request")
    return r.json(), _odds_api_usage_from_response(r)


@st.cache_data(ttl=90, show_spinner=False)
def fetch_sport_featured_odds(_api_key: str, sport_key: str, bookmakers_csv: str):
    if not _api_key:
        return [], {}
    url = f"{ODDS_API_BASE}/{sport_key}/odds"
    params = {
        "apiKey": _api_key,
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "dateFormat": "iso",
        "includeLinks": "true",
        "includeBetLimits": "true",
    }
    if bookmakers_csv:
        params["bookmakers"] = bookmakers_csv
    else:
        params["regions"] = "us"
    r = requests.get(url, params=params, timeout=40)
    _raise_market_api_error(r, f"{sport_key} featured odds request")
    return r.json(), _odds_api_usage_from_response(r)


@st.cache_data(ttl=90, show_spinner=False)
def fetch_sport_event_odds(_api_key: str, sport_key: str, event_id: str, markets_csv: str, bookmakers_csv: str):
    if not _api_key or not event_id or not markets_csv:
        return {}, {}
    url = f"{ODDS_API_BASE}/{sport_key}/events/{event_id}/odds"
    params = {
        "apiKey": _api_key,
        "markets": markets_csv,
        "oddsFormat": "american",
        "dateFormat": "iso",
        "includeLinks": "true",
        "includeBetLimits": "true",
    }
    if bookmakers_csv:
        params["bookmakers"] = bookmakers_csv
    else:
        params["regions"] = "us"
    r = requests.get(url, params=params, timeout=45)
    if r.status_code == 404:
        return {}, _odds_api_usage_from_response(r)
    _raise_market_api_error(r, f"{sport_key} event odds request")
    return r.json(), _odds_api_usage_from_response(r)


def fetch_sport_event_odds_live(_api_key: str, sport_key: str, event_id: str, markets_csv: str, bookmakers_csv: str):
    """Uncached one-market check used by the Market Radar inspector.

    This intentionally bypasses Streamlit's broad-scan cache so a user can verify
    an interesting quote immediately before acting on it.
    """
    if not _api_key or not event_id or not markets_csv:
        return {}, {}
    url = f"{ODDS_API_BASE}/{sport_key}/events/{event_id}/odds"
    params = {
        "apiKey": _api_key,
        "markets": markets_csv,
        "oddsFormat": "american",
        "dateFormat": "iso",
        "includeLinks": "true",
        "includeBetLimits": "true",
    }
    if bookmakers_csv:
        params["bookmakers"] = bookmakers_csv
    else:
        params["regions"] = "us"
    r = requests.get(url, params=params, timeout=45)
    if r.status_code == 404:
        return {}, _odds_api_usage_from_response(r)
    _raise_market_api_error(r, f"{sport_key} live verification request")
    return r.json(), _odds_api_usage_from_response(r)


@st.cache_data(ttl=180, show_spinner=False)
def fetch_futures_odds(_api_key: str, future_sport_key: str, bookmakers_csv: str):
    if not _api_key:
        return [], {}
    url = f"{ODDS_API_BASE}/{future_sport_key}/odds"
    params = {
        "apiKey": _api_key,
        "markets": "outrights",
        "oddsFormat": "american",
        "dateFormat": "iso",
        "includeLinks": "true",
        "includeBetLimits": "true",
    }
    if bookmakers_csv:
        params["bookmakers"] = bookmakers_csv
    else:
        params["regions"] = "us"
    r = requests.get(url, params=params, timeout=40)
    # An out-of-season future may return 404 or an empty response; neither should break the board.
    if r.status_code == 404:
        return [], _odds_api_usage_from_response(r)
    _raise_market_api_error(r, f"{future_sport_key} futures request")
    return r.json(), _odds_api_usage_from_response(r)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weather(lat: float, lon: float, venue_name: str, kickoff_utc_iso: str):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation_probability,wind_speed_10m,wind_gusts_10m",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "auto",
        "forecast_days": 16,
    }
    r = requests.get(WEATHER_URL, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    tz_name = payload.get("timezone") or "America/New_York"
    kickoff_utc = datetime.fromisoformat(kickoff_utc_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    local_dt = kickoff_utc.astimezone(ZoneInfo(tz_name))
    target = local_dt.replace(minute=0, second=0, microsecond=0)

    hourly = pd.DataFrame(payload["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    naive_target = target.replace(tzinfo=None)
    idx = (hourly["time"] - pd.Timestamp(naive_target)).abs().idxmin()
    row = hourly.loc[idx]

    return {
        "resolved_name": venue_name,
        "timezone": tz_name,
        "kickoff_local": local_dt,
        "temperature": float(row["temperature_2m"]),
        "wind": float(row["wind_speed_10m"]),
        "gust": float(row["wind_gusts_10m"]),
        "precip": float(row["precipitation_probability"]),
    }

# -----------------------------
# Football metrics
# -----------------------------
@st.cache_data(show_spinner=False)
def aggregate_game_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    df = pbp.copy()
    for c in ["pass", "rush", "epa", "yards_gained", "interception", "fumble_lost", "sack", "week"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[
        (df["season_type"] == "REG")
        & df["posteam"].notna()
        & df["defteam"].notna()
        & df["epa"].notna()
        & ((df["pass"] == 1) | (df["rush"] == 1))
    ].copy()

    df["success"] = (df["epa"] > 0).astype(float)
    df["explosive"] = (df["yards_gained"].fillna(0) >= 20).astype(float)
    df["turnover"] = (
        (df["interception"].fillna(0) == 1) | (df["fumble_lost"].fillna(0) == 1)
    ).astype(float)
    df["pass_epa_value"] = df["epa"].where(df["pass"] == 1)
    df["rush_epa_value"] = df["epa"].where(df["rush"] == 1)

    g = (
        df.groupby(["game_id", "posteam", "defteam", "week"], as_index=False)
        .agg(
            epa=("epa", "mean"),
            pass_epa=("pass_epa_value", "mean"),
            rush_epa=("rush_epa_value", "mean"),
            success=("success", "mean"),
            explosive=("explosive", "mean"),
            plays=("epa", "size"),
            turnovers=("turnover", "sum"),
            pass_plays=("pass", "sum"),
            sacks=("sack", "sum"),
        )
    )
    g["turnover_rate"] = g["turnovers"] / g["plays"].clip(lower=1)
    g["sack_rate"] = g["sacks"] / g["pass_plays"].clip(lower=1)
    return g

def build_histories(metrics: pd.DataFrame, max_week: int | None = None):
    m = metrics.copy()
    if max_week is not None:
        m = m[m["week"] < max_week]
    m = m.sort_values(["week", "game_id"])
    states: dict[str, dict[str, list[dict]]] = {}

    for _, game in m.groupby("game_id", sort=False):
        rows = list(game.to_dict("records"))
        if len(rows) < 2:
            continue
        by_team = {r["posteam"]: r for r in rows}
        for r in rows:
            team = r["posteam"]
            opp = r["defteam"]
            if opp not in by_team:
                continue
            states.setdefault(team, {"off": [], "def": []})
            metric_keys = ["epa", "pass_epa", "rush_epa", "success", "explosive", "plays", "turnover_rate", "sack_rate"]
            off = {k: r.get(k) for k in metric_keys}
            deff = {k: by_team[opp].get(k) for k in metric_keys}
            states[team]["off"].append(off)
            states[team]["def"].append(deff)
    return states

def mean_valid(values):
    vals = [float(v) for v in values if v is not None and not pd.isna(v)]
    return sum(vals) / len(vals) if vals else None

def blended_metric(history: list[dict], metric: str, season_weight=0.65, last4_weight=0.35):
    if not history:
        return None
    season = mean_valid([x.get(metric) for x in history])
    recent = mean_valid([x.get(metric) for x in history[-4:]])
    if season is None:
        return recent
    if recent is None:
        return season
    return season_weight * season + last4_weight * recent

def combine_state(prev_states, cur_states, team, strict: bool):
    current = cur_states.get(team, {"off": [], "def": []})
    if strict or len(current["off"]) >= 4:
        return current, False

    previous = prev_states.get(team, {"off": [], "def": []})
    # Experimental early-season carryover: use enough of the previous season's final
    # four games to reach at least four pregame games.
    needed = max(0, 4 - len(current["off"]))
    off = previous["off"][-needed:] + current["off"] if needed else current["off"]
    deff = previous["def"][-needed:] + current["def"] if needed else current["def"]
    return {"off": off, "def": deff}, needed > 0

def matchup_features(away: str, home: str, prev_states, cur_states, strict: bool):
    a, a_prior = combine_state(prev_states, cur_states, away, strict)
    h, h_prior = combine_state(prev_states, cur_states, home, strict)

    if len(a["off"]) < 4 or len(h["off"]) < 4:
        return None, {
            "away_games": len(cur_states.get(away, {"off": []})["off"]),
            "home_games": len(cur_states.get(home, {"off": []})["off"]),
            "used_prior": False,
        }

    def metric(state, side, name):
        return blended_metric(state[side], name)

    def mu(*values):
        x = mean_valid(values)
        return 0.0 if x is None else x

    def matchup(name):
        away_expected = mu(metric(a, "off", name), metric(h, "def", name))
        home_expected = mu(metric(h, "off", name), metric(a, "def", name))
        return away_expected + home_expected

    f = {
        "EPA_Matchup": matchup("epa"),
        "Pass_EPA_Matchup": matchup("pass_epa"),
        "Rush_EPA_Matchup": matchup("rush_epa"),
        "Success_Matchup": matchup("success"),
        "Explosive_Matchup": matchup("explosive"),
        "Plays_Matchup": mu(metric(a, "off", "plays")) + mu(metric(h, "off", "plays")),
        "Turnover_Env": matchup("turnover_rate"),
        "Sack_Env": matchup("sack_rate"),
    }
    meta = {
        "away_games": len(cur_states.get(away, {"off": []})["off"]),
        "home_games": len(cur_states.get(home, {"off": []})["off"]),
        "used_prior": a_prior or h_prior,
    }
    return f, meta

# -----------------------------
# Odds and model helpers
# -----------------------------
def to_float(v, default=None):
    try:
        x = float(v)
        if math.isnan(x):
            return default
        return x
    except Exception:
        return default

def season_from_kickoff(kickoff: datetime) -> int:
    return kickoff.year if kickoff.month >= 7 else kickoff.year - 1

def schedule_kickoff_utc(game_row) -> datetime:
    gameday = pd.Timestamp(game_row["gameday"]).date()
    gt = str(game_row.get("gametime") or "13:00")
    try:
        hh, mm = [int(x) for x in gt.split(":")[:2]]
    except Exception:
        hh, mm = 13, 0
    eastern = datetime(gameday.year, gameday.month, gameday.day, hh, mm, tzinfo=ZoneInfo("America/New_York"))
    return eastern.astimezone(timezone.utc)

def upcoming_schedule(schedule: pd.DataFrame, days_ahead=10):
    today = pd.Timestamp(date.today())
    end = today + pd.Timedelta(days=days_ahead)
    x = schedule[
        (schedule["game_type"] == "REG")
        & (schedule["gameday"] >= today)
        & (schedule["gameday"] <= end)
    ].copy()
    return x.sort_values(["gameday", "gametime", "game_id"])

def match_odds_event(odds_events, away: str, home: str):
    for e in odds_events:
        if ODDS_NAME_TO_ABBR.get(e.get("away_team")) == away and ODDS_NAME_TO_ABBR.get(e.get("home_team")) == home:
            return e
    return None

def book_market_rows(event):
    rows = []
    if not event:
        return rows

    home_name = event.get("home_team")
    away_name = event.get("away_team")

    for b in event.get("bookmakers", []):
        markets = b.get("markets", [])
        total_market = next((m for m in markets if m.get("key") == "totals"), None)
        spread_market = next((m for m in markets if m.get("key") == "spreads"), None)
        h2h_market = next((m for m in markets if m.get("key") == "h2h"), None)

        over = under = None
        if total_market:
            over = next((o for o in total_market.get("outcomes", []) if o.get("name") == "Over"), None)
            under = next((o for o in total_market.get("outcomes", []) if o.get("name") == "Under"), None)

        home_spread_out = away_spread_out = None
        if spread_market:
            home_spread_out = next(
                (o for o in spread_market.get("outcomes", []) if o.get("name") == home_name), None
            )
            away_spread_out = next(
                (o for o in spread_market.get("outcomes", []) if o.get("name") == away_name), None
            )

        home_ml_out = away_ml_out = None
        if h2h_market:
            home_ml_out = next(
                (o for o in h2h_market.get("outcomes", []) if o.get("name") == home_name), None
            )
            away_ml_out = next(
                (o for o in h2h_market.get("outcomes", []) if o.get("name") == away_name), None
            )

        # Keep any bookmaker that has at least one of our three core markets.
        if not any([total_market, spread_market, h2h_market]):
            continue

        rows.append({
            "key": b.get("key"),
            "book": b.get("title", b.get("key")),
            "total": to_float(over.get("point")) if over else None,
            "over": to_float(over.get("price")) if over else None,
            "under": to_float(under.get("price")) if under else None,
            "home_spread": to_float(home_spread_out.get("point")) if home_spread_out else None,
            "away_spread": to_float(away_spread_out.get("point")) if away_spread_out else None,
            "home_spread_odds": to_float(home_spread_out.get("price")) if home_spread_out else None,
            "away_spread_odds": to_float(away_spread_out.get("price")) if away_spread_out else None,
            "home_ml": to_float(home_ml_out.get("price")) if home_ml_out else None,
            "away_ml": to_float(away_ml_out.get("price")) if away_ml_out else None,
            "updated": b.get("last_update"),
        })
    return rows

def american_profit(odds):
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)

def implied_prob_american(odds):
    if odds is None:
        return None
    odds = float(odds)
    if odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)

def devig_pair(odds_a, odds_b):
    pa = implied_prob_american(odds_a)
    pb = implied_prob_american(odds_b)
    if pa is None or pb is None or (pa + pb) <= 0:
        return None, None
    total = pa + pb
    return pa / total, pb / total

def ev_from_prob(prob, odds):
    if prob is None or odds is None:
        return None
    return prob * american_profit(float(odds)) - (1.0 - prob)

def consensus_moneyline(rows, selected_key=None):
    usable = [
        r for r in rows
        if r.get("home_ml") is not None and r.get("away_ml") is not None
        and r.get("key") != selected_key
    ]
    used_selected = False
    if not usable:
        usable = [
            r for r in rows
            if r.get("home_ml") is not None and r.get("away_ml") is not None
        ]
        used_selected = True

    probs = []
    for r in usable:
        ph, pa = devig_pair(r["home_ml"], r["away_ml"])
        if ph is not None:
            probs.append((ph, pa))

    if not probs:
        return None, None, 0, used_selected

    home_p = sum(x[0] for x in probs) / len(probs)
    away_p = 1.0 - home_p
    return home_p, away_p, len(probs), used_selected

def consensus_spread(rows, target_home_spread, selected_key=None):
    def same_line(r):
        x = r.get("home_spread")
        return (
            x is not None
            and target_home_spread is not None
            and abs(float(x) - float(target_home_spread)) < 0.01
            and r.get("home_spread_odds") is not None
            and r.get("away_spread_odds") is not None
        )

    usable = [r for r in rows if same_line(r) and r.get("key") != selected_key]
    used_selected = False
    if not usable:
        usable = [r for r in rows if same_line(r)]
        used_selected = True

    probs = []
    for r in usable:
        ph, pa = devig_pair(r["home_spread_odds"], r["away_spread_odds"])
        if ph is not None:
            probs.append((ph, pa))

    if not probs:
        return None, None, 0, used_selected

    home_p = sum(x[0] for x in probs) / len(probs)
    away_p = 1.0 - home_p
    return home_p, away_p, len(probs), used_selected

def fair_american(p):
    if not (0 < p < 1):
        return None
    return -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p

def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def model_output(features, total, over_odds, under_odds):
    correction = float(COEF["Intercept"])
    for f in FEATURES:
        correction += float(COEF[f]) * float(features[f])
    model_total = total + correction
    sd = float(COEF["Residual_SD"])
    p_over = normal_cdf(correction / sd)
    p_under = 1 - p_over
    over_ev = p_over * american_profit(over_odds) - (1 - p_over)
    under_ev = p_under * american_profit(under_odds) - (1 - p_under)

    v1_signal = "PASS"
    if abs(correction) >= 2:
        if correction > 0 and p_over >= 0.54 and over_ev >= 0.025:
            v1_signal = "OVER"
        elif correction < 0 and p_under >= 0.54 and under_ev >= 0.025:
            v1_signal = "UNDER"

    return {
        "correction": correction,
        "model_total": model_total,
        "p_over": p_over,
        "p_under": p_under,
        "fair_over": fair_american(p_over),
        "fair_under": fair_american(p_under),
        "over_ev": over_ev,
        "under_ev": under_ev,
        "signal": v1_signal,
    }

def fmt_odds(x):
    if x is None:
        return "—"
    x = round(float(x))
    return f"+{x}" if x > 0 else str(x)


# -----------------------------
# Weekly model favorites
# -----------------------------
def _weekly_reference_market(book_rows, game_row):
    """Use DraftKings when available; otherwise the most complete live book, then nflverse fallbacks."""
    row = None
    if book_rows:
        core = [
            "total", "over", "under", "home_spread",
            "home_spread_odds", "away_spread_odds", "home_ml", "away_ml"
        ]
        # Prefer the most complete market snapshot. DraftKings wins ties so the
        # weekly board generally matches the single-game app's default book.
        candidates = sorted(
            book_rows,
            key=lambda r: (sum(r.get(k) is not None for k in core), 1 if r.get("key") == "draftkings" else 0),
            reverse=True,
        )
        row = dict(candidates[0]) if candidates else None

    fallback_total = to_float(game_row.get("total_line"), 44.5)
    fallback_spread = -to_float(game_row.get("spread_line"), 0.0)
    fallback_home_ml = to_float(game_row.get("home_moneyline"), None)
    fallback_away_ml = to_float(game_row.get("away_moneyline"), None)
    out = {
        "book": (row or {}).get("book", "nflverse / fallback"),
        "total": (row or {}).get("total") if (row or {}).get("total") is not None else fallback_total,
        "over": (row or {}).get("over") if (row or {}).get("over") is not None else -110.0,
        "under": (row or {}).get("under") if (row or {}).get("under") is not None else -110.0,
        "home_spread": (row or {}).get("home_spread") if (row or {}).get("home_spread") is not None else fallback_spread,
        "home_spread_odds": (row or {}).get("home_spread_odds") if (row or {}).get("home_spread_odds") is not None else -110.0,
        "away_spread_odds": (row or {}).get("away_spread_odds") if (row or {}).get("away_spread_odds") is not None else -110.0,
        "home_ml": (row or {}).get("home_ml") if (row or {}).get("home_ml") is not None else fallback_home_ml,
        "away_ml": (row or {}).get("away_ml") if (row or {}).get("away_ml") is not None else fallback_away_ml,
    }
    return out


def _weekly_weather_inputs(game_row, home_team):
    roof_type = "fixed" if home_team in FIXED_ROOF else "retractable" if home_team in RETRACTABLE_ROOF else "outdoor"
    schedule_roof = str(game_row.get("roof") or "").lower()
    if roof_type == "fixed":
        return 70.0, 0.0, 1.0, "fixed roof"
    if roof_type == "retractable" and schedule_roof in {"closed", "dome"}:
        return 70.0, 0.0, 1.0, "closed roof"

    temp, wind = 70.0, 5.0
    label = "forecast fallback"
    loc = WEATHER_COORDS.get(home_team)
    if loc:
        try:
            lat, lon, venue_name = loc
            wx = fetch_weather(lat, lon, venue_name, schedule_kickoff_utc(game_row).isoformat())
            temp = float(wx["temperature"])
            wind = float(wx["wind"])
            label = "stadium forecast"
        except Exception:
            pass
    return temp, wind, 0.0, label


def build_weekly_favorites_board(
    *, schedule, week_games, odds_events, prev_pbp, cur_pbp, prev_states, cur_states,
    season, week, strict_mode, root
):
    """Rank picks by model confidence, not by EV. V2.5 handles sides when available; V1 handles totals."""
    ml_rows, spread_rows, total_rows, skipped = [], [], [], []
    side_builder = build_side_prediction_v25 if build_side_prediction_v25 is not None else build_side_prediction_v24
    side_model_label = "V2.5" if build_side_prediction_v25 is not None else "V2.4"
    if side_builder is None:
        return {"ml": pd.DataFrame(), "spread": pd.DataFrame(), "total": pd.DataFrame(), "skipped": ["Side model unavailable"], "games_scanned": 0, "side_model": "unavailable"}

    for _, g in week_games.sort_values(["gameday", "gametime", "game_id"]).iterrows():
        away_t, home_t = str(g["away_team"]), str(g["home_team"])
        matchup = f"{away_t} @ {home_t}"
        rows = book_market_rows(match_odds_event(odds_events, away_t, home_t)) if odds_events else []
        mkt = _weekly_reference_market(rows, g)

        # Side model requires moneyline prices for its calibrated ML probability.
        if mkt["home_ml"] is None or mkt["away_ml"] is None:
            skipped.append(f"{matchup} (moneyline unavailable)")
            continue

        ff, _ = matchup_features(away_t, home_t, prev_states, cur_states, strict=strict_mode)
        if ff is None:
            skipped.append(f"{matchup} (team ratings unavailable)")
            continue

        temp, wind, dome_i, weather_src = _weekly_weather_inputs(g, home_t)
        away_rest_i = to_float(g.get("away_rest"), 7.0)
        home_rest_i = to_float(g.get("home_rest"), 7.0)
        short_rest_i = 1.0 if min(away_rest_i, home_rest_i) <= 5 else 0.0
        try:
            div_i = 1.0 if float(g.get("div_game")) == 1 else 0.0
        except Exception:
            div_i = 0.0
        cold_i = 0.0 if dome_i == 1.0 else max(0.0, (40.0 - temp) / 10.0)
        total_features = dict(ff)
        total_features.update({
            "Wind": float(wind), "Cold_Units": float(cold_i), "Dome": float(dome_i),
            "Short_Rest": short_rest_i, "Div_Game": div_i,
        })
        total_res = model_output(
            total_features, float(mkt["total"]), float(mkt["over"]), float(mkt["under"])
        )

        try:
            side = side_builder(
                root=root, schedule=schedule, game=g, prev_pbp=prev_pbp, cur_pbp=cur_pbp,
                home=home_t, away=away_t, season=int(season), week=int(week),
                market_total=float(mkt["total"]), over_odds=float(mkt["over"]), under_odds=float(mkt["under"]),
                home_spread=float(mkt["home_spread"]),
                home_spread_odds=float(mkt["home_spread_odds"]), away_spread_odds=float(mkt["away_spread_odds"]),
                home_ml=float(mkt["home_ml"]), away_ml=float(mkt["away_ml"]), dome=float(dome_i),
            )
        except Exception as exc:
            skipped.append(f"{matchup} ({side_model_label}: {exc})")
            continue

        # Moneyline favorite = side with the larger side-model win probability. Rank is confidence only.
        if float(side["p_home_win"]) >= 0.5:
            ml_pick, ml_p, ml_price, ml_fair, ml_ev = home_t, float(side["p_home_win"]), mkt["home_ml"], side["fair_home_ml"], side["home_ml_ev"]
        else:
            ml_pick, ml_p, ml_price, ml_fair, ml_ev = away_t, float(side["p_away_win"]), mkt["away_ml"], side["fair_away_ml"], side["away_ml_ev"]
        ml_rows.append({
            "Matchup": matchup, "Pick": f"{ml_pick} ML", "Confidence": ml_p,
            "Price": ml_price, "Fair": ml_fair, "EV": float(ml_ev),
            "Side margin": float(side["predicted_margin"]), "Book": mkt["book"],
        })

        # Spread favorite = side with the larger side-model cover probability.
        if float(side["p_home_cover"]) >= 0.5:
            sp_pick, sp_line, sp_p, sp_price, sp_ev = home_t, float(mkt["home_spread"]), float(side["p_home_cover"]), mkt["home_spread_odds"], side["home_spread_ev"]
        else:
            sp_pick, sp_line, sp_p, sp_price, sp_ev = away_t, -float(mkt["home_spread"]), float(side["p_away_cover"]), mkt["away_spread_odds"], side["away_spread_ev"]
        market_home_margin = -float(mkt["home_spread"])
        spread_rows.append({
            "Matchup": matchup, "Pick": f"{sp_pick} {sp_line:+.1f}", "Confidence": sp_p,
            "Price": sp_price, "EV": float(sp_ev), "Model margin edge": abs(float(side["predicted_margin"]) - market_home_margin),
            "Book": mkt["book"],
        })

        # Totals favorite = higher V1 O/U probability. It is intentionally shown as experimental.
        if float(total_res["p_over"]) >= 0.5:
            tot_pick, tot_p, tot_price, tot_ev = "OVER", float(total_res["p_over"]), mkt["over"], total_res["over_ev"]
        else:
            tot_pick, tot_p, tot_price, tot_ev = "UNDER", float(total_res["p_under"]), mkt["under"], total_res["under_ev"]
        total_rows.append({
            "Matchup": matchup, "Pick": f"{tot_pick} {float(mkt['total']):.1f}", "Confidence": tot_p,
            "Price": tot_price, "EV": float(tot_ev), "V1 model total": float(total_res["model_total"]),
            "Model edge": abs(float(total_res["model_total"]) - float(mkt["total"])),
            "Weather": weather_src, "Book": mkt["book"],
        })

    def top3(rows, secondary):
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df.sort_values(["Confidence", secondary], ascending=[False, False]).head(3).reset_index(drop=True)

    return {
        "ml": top3(ml_rows, "EV"),
        "spread": top3(spread_rows, "Model margin edge"),
        "total": top3(total_rows, "Model edge"),
        "skipped": skipped,
        "games_scanned": len(ml_rows),
        "side_model": side_model_label,
    }


def render_weekly_favorites(board, week):
    st.subheader(f"⭐ Weekly Model Favorites — Week {int(week)}")
    st.caption(
        f"These are ranked by MODEL CONFIDENCE, not by expected value. Moneyline and spread use {board.get('side_model','V2.5')}; totals use the experimental V1 totals model. "
        "EV and prices are shown only as context, so a top-ranked pick can still be a bad price."
    )
    if not board or board.get("games_scanned", 0) == 0:
        st.warning("No complete weekly board could be built from the currently available markets/data.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Games scanned", board.get("games_scanned", 0))
    c2.metric("Side model", board.get("side_model", "V2.5"))
    c3.metric("Totals model", "V1 experimental")

    def prep(df, kind):
        if df is None or df.empty:
            return df
        z = df.copy()
        z.insert(0, "Rank", range(1, len(z) + 1))
        z["Confidence"] = z["Confidence"].map(lambda x: f"{float(x):.1%}")
        if "EV" in z:
            z["EV"] = z["EV"].map(lambda x: f"{float(x):+.1%}")
        if "Price" in z:
            z["Price"] = z["Price"].map(fmt_odds)
        if "Fair" in z:
            z["Fair"] = z["Fair"].map(fmt_odds)
        if "Side margin" in z:
            z["Side margin"] = z["Side margin"].map(lambda x: f"{float(x):+.1f}")
        if "Model margin edge" in z:
            z["Model margin edge"] = z["Model margin edge"].map(lambda x: f"{float(x):.1f} pts")
        if "V1 model total" in z:
            z["V1 model total"] = z["V1 model total"].map(lambda x: f"{float(x):.1f}")
        if "Model edge" in z:
            z["Model edge"] = z["Model edge"].map(lambda x: f"{float(x):.1f} pts")
        return z

    st.markdown("#### Top 3 Moneylines")
    st.dataframe(prep(board["ml"], "ml"), width="stretch", hide_index=True)
    st.markdown("#### Top 3 Spreads")
    st.dataframe(prep(board["spread"], "spread"), width="stretch", hide_index=True)
    st.markdown("#### Top 3 Totals")
    st.dataframe(prep(board["total"], "total"), width="stretch", hide_index=True)
    st.warning("Totals are shown because you asked for them, but V1 totals lost 18.2% ROI in the 2025 out-of-sample backtest. Treat the totals list as R&D, not a validated betting recommendation.")

    if board.get("skipped"):
        with st.expander(f"Skipped / incomplete games ({len(board['skipped'])})"):
            for x in board["skipped"]:
                st.write("•", x)

# -----------------------------
# Multi-sport Market Radar
# -----------------------------
def _parse_utc_timestamp(value):
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts
    except Exception:
        return None


def _radar_target_events(events, horizon_hours: int, max_events: int):
    now = pd.Timestamp.now(tz="UTC")
    cutoff = now + pd.Timedelta(hours=int(horizon_hours))
    good = []
    for ev in events or []:
        ts = _parse_utc_timestamp(ev.get("commence_time"))
        if ts is None:
            continue
        # Keep live/recently-started games because some books still expose markets,
        # but do not include events that began more than 3 hours ago.
        if now - pd.Timedelta(hours=3) <= ts <= cutoff:
            good.append(ev)
    good.sort(key=lambda x: str(x.get("commence_time") or ""))
    return good[: int(max_events)]


def _radar_market_keys(league: str, scan_mode: str, include_extra: bool):
    keys = []
    if scan_mode != "Game lines only":
        if scan_mode == "Smart props":
            keys.extend(SPORT_CORE_PROP_KEYS.get(league, []))
        else:
            keys.extend(list(SPORT_PROP_MARKETS.get(league, {}).keys()))
        if scan_mode == "Deep + alternates":
            keys.extend(SPORT_ALTERNATE_KEYS.get(league, []))
    if include_extra:
        keys.extend(EXTRA_GAME_MARKETS.get(league, []))
    # Preserve order, remove duplicates.
    return list(dict.fromkeys(keys))


def _split_watch_terms(text):
    if not text:
        return []
    cleaned = str(text).replace(",", "\n")
    return [x.strip() for x in cleaned.splitlines() if x.strip()]


def _radar_type_icon(kind):
    k = str(kind)
    if k.startswith("TRUE ARB"):
        return "🟢 " + k
    if k == "ARB + MIDDLE":
        return "🟢🟣 " + k
    if k == "MIDDLE":
        return "🟣 " + k
    if k == "LINE OUTLIER":
        return "🔵 " + k
    if k == "EXCHANGE GAP":
        return "🟡 " + k
    if k == "PRICE OUTLIER":
        return "🟠 " + k
    return k


def _quote_age_text(value):
    ts = _parse_utc_timestamp(value)
    if ts is None:
        return "—"
    age = max(0.0, (pd.Timestamp.now(tz="UTC") - ts).total_seconds())
    if age < 60:
        return f"{int(age)}s"
    if age < 3600:
        return f"{int(age // 60)}m {int(age % 60)}s"
    return f"{age / 3600:.1f}h"


def _balanced_radar_view(df: pd.DataFrame, max_rows: int = 100) -> pd.DataFrame:
    """Diversify the default board so one noisy longshot market cannot own the screen.

    True arbs/middles keep their full priority. TD/first-scorer style price gaps stay
    available, but receive a display penalty and a row cap. The raw board remains
    available through the All signals view.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    x = df.copy()
    if "Market Group" not in x.columns:
        x["Market Group"] = "Other markets"
    x["_rank"] = pd.to_numeric(x.get("Score"), errors="coerce").fillna(0.0)
    arbish = x["Type"].astype(str).str.startswith("TRUE ARB") | x["Type"].astype(str).isin(["ARB + MIDDLE", "MIDDLE"])
    scorer = x["Market Group"].astype(str).eq("TD / scorer props") & ~arbish
    # Longshot relative-payout gaps are useful, but should not crowd out yardage,
    # receptions, game lines, or actual arbs.
    x.loc[scorer, "_rank"] = x.loc[scorer, "_rank"] - 14.0
    x = x.sort_values(["_rank", "Score"], ascending=[False, False], na_position="last")

    kept = []
    scorer_count = 0
    per_market = {}
    per_matchup = {}
    for idx, row in x.iterrows():
        group = str(row.get("Market Group", ""))
        market = str(row.get("Market Key", row.get("Market", "")))
        matchup = str(row.get("Matchup", ""))
        is_arbish = str(row.get("Type", "")).startswith("TRUE ARB") or str(row.get("Type", "")) in {"ARB + MIDDLE", "MIDDLE"}
        if group == "TD / scorer props" and not is_arbish and scorer_count >= 10:
            continue
        if not is_arbish and per_market.get(market, 0) >= 8:
            continue
        if not is_arbish and per_matchup.get(matchup, 0) >= 12:
            continue
        kept.append(idx)
        per_market[market] = per_market.get(market, 0) + 1
        per_matchup[matchup] = per_matchup.get(matchup, 0) + 1
        if group == "TD / scorer props" and not is_arbish:
            scorer_count += 1
        if len(kept) >= int(max_rows):
            break
    return x.loc[kept].drop(columns=["_rank"], errors="ignore").reset_index(drop=True)


def render_market_radar(api_key: str, selected_book_keys, bookmaker_csv: str):
    st.markdown("---")
    st.subheader("🔎 Market Radar — automatic mispricing scanner")
    st.caption(
        "This is the Pikkit-style screen that does the scrolling for you. It scans only your selected books, then surfaces "
        "true arbs, middles, unusually good lines, exact-price outliers and sportsbook-vs-exchange disagreements. "
        "The default Balanced feed prevents one longshot market (like first-TD scorers) from taking over the board. "
        "A flag is a market anomaly, not proof that a side will win."
    )
    with st.expander("What each flag means", expanded=False):
        st.markdown(
            "**TRUE ARB** — opposing prices mathematically lock a positive return under the displayed fee assumptions.  \n"
            "**ARB + MIDDLE** — the prices form an arb and the line gap also creates a range where both bets can win.  \n"
            "**MIDDLE** — favorable line gap, but not guaranteed profit; both bets can still lose outside the middle.  \n"
            "**LINE OUTLIER** — one selected book gives a materially better number than the others.  \n"
            "**PRICE OUTLIER** — the exact same selection/line pays materially more at one book.  \n"
            "**EXCHANGE GAP** — the exact same selection is priced unusually differently between an exchange and the rest of your market."
        )
        st.caption("The 0–100 Mispricing Score ranks how unusual/useful the discrepancy looks. It is not a predicted win probability or EV estimate.")
    st.info(
        "Championship futures are supported when the feed has them. Team season-win totals (for example, Steelers O/U 8.5 wins) "
        "are not currently a documented standard Odds API market, so this version does not pretend to scan those automatically yet."
    )

    if not api_key:
        st.info("Add The Odds API key in the sidebar to use Market Radar.")
        return
    if not selected_book_keys:
        st.info("Select at least two sportsbooks/exchanges in the sidebar.")
        return

    top_a, top_b, top_c, top_d = st.columns([1.1, 1.1, 1.1, 1.2])
    leagues = top_a.multiselect(
        "Leagues", ["NFL", "NBA", "MLB", "NHL"], default=["NFL", "NBA", "MLB", "NHL"], key="radar_leagues"
    )
    scan_mode = top_b.selectbox(
        "Scan depth", ["Game lines only", "Smart props", "Deep props", "Deep + alternates"],
        index=1, key="radar_depth",
        help="Smart props scans the most useful props. Deep adds nearly every listed main player prop. Alternates can use many more credits."
    )
    horizon_label = top_c.selectbox(
        "Games starting within", ["12 hours", "24 hours", "48 hours", "72 hours", "7 days"],
        index=2, key="radar_horizon"
    )
    max_events = top_d.selectbox("Max events per league", [5, 10, 16, 20, 30], index=2, key="radar_max_events")
    horizon_hours = {"12 hours": 12, "24 hours": 24, "48 hours": 48, "72 hours": 72, "7 days": 168}[horizon_label]

    opt1, opt2, opt3, opt4, opt5 = st.columns(5)
    include_extra = opt1.checkbox(
        "Extra game markets", value=False, key="radar_extra",
        help="Adds team totals / 1st half / first 5 innings / first period where supported. These are event-level markets and use extra credits."
    )
    include_futures = opt2.checkbox("Championship futures", value=True, key="radar_futures")
    sensitivity = opt3.selectbox("Sensitivity", ["Very sensitive", "Balanced", "Strict"], index=0, key="radar_sensitivity")
    arb_bankroll = opt4.number_input("Arb example stake ($)", min_value=10.0, value=100.0, step=25.0, key="radar_arb_stake")
    exchange_fee_buffer = opt5.number_input(
        "Exchange fee buffer %", min_value=0.0, max_value=10.0, value=1.0, step=0.25, key="radar_exchange_fee_buffer",
        help="Conservative generic haircut on the profit portion of exchange prices when testing arbs/price gaps. Actual Novig/Kalshi/Polymarket/ProphetX fees and liquidity differ, so verify the executable price and fee before betting."
    ) / 100.0

    if sensitivity == "Very sensitive":
        min_price_adv, min_line_adv = 0.010, 0.5
    elif sensitivity == "Balanced":
        min_price_adv, min_line_adv = 0.025, 0.5
    else:
        min_price_adv, min_line_adv = 0.050, 1.0

    with st.expander("⭐ Sides / players I already like", expanded=False):
        watch_text = st.text_area(
            "One interest per line (or comma separated)",
            value=st.session_state.get("radar_watch_text", ""),
            placeholder="Josh Allen over\nSteelers\nPuka receiving over",
            key="radar_watch_input",
            help="These do not change the math. They only put a star on matching anomalies so a better number on a side you already like is easy to notice."
        )
        st.session_state["radar_watch_text"] = watch_text
        st.caption("Example: `Josh Allen over` will prioritize matching Josh Allen Over opportunities without calling them +EV just because you like the side.")
    watch_terms = _split_watch_terms(st.session_state.get("radar_watch_text", ""))

    # Events are free to query, so use them to estimate the scan before the user spends credits.
    target_events_by_league = {}
    discovery_errors = []
    for league in leagues:
        try:
            events, usage = fetch_sport_events(api_key, SPORT_KEYS[league])
            remember_odds_api_usage(usage)
            target_events_by_league[league] = _radar_target_events(events, horizon_hours, max_events)
        except Exception as exc:
            discovery_errors.append(f"{league}: {exc}")
            target_events_by_league[league] = []

    groups = max(1, math.ceil(max(1, len(selected_book_keys)) / 10))
    featured_cost = len(leagues) * 3 * groups
    event_cost = 0
    event_calls = 0
    for league in leagues:
        event_markets = _radar_market_keys(league, scan_mode, include_extra)
        n_events = len(target_events_by_league.get(league, []))
        event_cost += n_events * len(event_markets) * groups
        if event_markets:
            event_calls += n_events
    futures_cost = (len(leagues) * groups) if include_futures else 0
    estimated_cost = featured_cost + event_cost + futures_cost

    counts_text = " · ".join(f"{lg}: {len(target_events_by_league.get(lg, []))}" for lg in leagues) if leagues else "No leagues selected"
    est_box = st.container(border=True)
    with est_box:
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Events in scope", sum(len(v) for v in target_events_by_league.values()))
        e2.metric("Max credit estimate", estimated_cost)
        usage_now = current_odds_api_usage()
        remaining = usage_now.get("remaining")
        e3.metric("Credits remaining", f"{int(remaining):,}" if remaining is not None else "—")
        e4.metric("Books selected", len(selected_book_keys))
        st.caption(f"Events discovered: {counts_text}. Actual cost can be lower because The Odds API charges event calls for unique markets actually returned.")

    if "williamhill_us" in selected_book_keys:
        st.warning("Caesars is currently a paid-only feed in The Odds API. If you are on the free plan and a scan returns a 422 error, uncheck Caesars in the sidebar and rerun.")

    if discovery_errors:
        with st.expander(f"Event discovery warnings ({len(discovery_errors)})"):
            for x in discovery_errors:
                st.write("•", x)

    expensive = estimated_cost >= 100
    confirm_expensive = True
    if expensive:
        confirm_expensive = st.checkbox(
            f"I understand this scan could use up to about {estimated_cost} credits",
            value=False, key="radar_confirm_expensive"
        )
        st.warning("This is a broad scan. Reduce the horizon, leagues or scan depth if you want to conserve the free monthly quota.")

    scan_disabled = (
        not leagues
        or len(selected_book_keys) < 2
        or (remaining is not None and estimated_cost > int(remaining))
        or not confirm_expensive
    )
    if remaining is not None and estimated_cost > int(remaining):
        st.error("The estimated maximum scan cost is greater than your remaining API credits. Narrow the scan first.")

    if st.button("🚨 Run Market Radar", type="primary", key="run_market_radar", disabled=scan_disabled):
        offer_frames = []
        errors = []
        total_calls = len(leagues) + event_calls + (len(leagues) if include_futures else 0)
        done_calls = 0
        progress = st.progress(0.0, text="Scanning markets…")

        for league in leagues:
            sport_key = SPORT_KEYS[league]
            try:
                payload, usage = fetch_sport_featured_odds(api_key, sport_key, bookmaker_csv)
                remember_odds_api_usage(usage)
                frame = normalize_market_payloads(payload, league)
                if not frame.empty:
                    offer_frames.append(frame)
            except Exception as exc:
                errors.append(f"{league} game lines: {exc}")
            done_calls += 1
            progress.progress(done_calls / max(1, total_calls), text=f"{league}: game lines")

            event_markets = _radar_market_keys(league, scan_mode, include_extra)
            if event_markets:
                market_csv = ",".join(event_markets)
                for ev in target_events_by_league.get(league, []):
                    try:
                        payload, usage = fetch_sport_event_odds(api_key, sport_key, str(ev.get("id")), market_csv, bookmaker_csv)
                        remember_odds_api_usage(usage)
                        frame = normalize_market_payloads(payload, league)
                        if not frame.empty:
                            offer_frames.append(frame)
                    except Exception as exc:
                        errors.append(f"{league} {ev.get('away_team')} @ {ev.get('home_team')}: {exc}")
                    done_calls += 1
                    progress.progress(done_calls / max(1, total_calls), text=f"{league}: props / extra markets")

            if include_futures:
                try:
                    payload, usage = fetch_futures_odds(api_key, FUTURE_SPORT_KEYS[league], bookmaker_csv)
                    remember_odds_api_usage(usage)
                    frame = normalize_market_payloads(payload, league)
                    if not frame.empty:
                        offer_frames.append(frame)
                except Exception as exc:
                    # Futures can simply be out of season; keep the error available but do not stop the scan.
                    errors.append(f"{league} futures: {exc}")
                done_calls += 1
                progress.progress(done_calls / max(1, total_calls), text=f"{league}: futures")

        progress.empty()
        offers = combine_normalized_offer_frames(offer_frames)
        board = build_opportunity_board(
            offers,
            total_stake=float(arb_bankroll),
            min_payout_advantage=min_price_adv,
            min_line_advantage=min_line_adv,
            min_books=2,
            include_futures_arb=include_futures,
            exchange_fee_buffer=float(exchange_fee_buffer),
        )
        board = apply_watchlist(board, watch_terms)
        st.session_state["market_radar_scan"] = {
            "offers": offers,
            "board": board,
            "errors": errors,
            "leagues": leagues,
            "book_keys": list(selected_book_keys),
            "estimated_cost": estimated_cost,
            "scan_mode": scan_mode,
            "horizon": horizon_label,
            "exchange_fee_buffer": float(exchange_fee_buffer),
            "scanned_at": datetime.now(timezone.utc).isoformat(),
        }
        st.rerun()

    scan = st.session_state.get("market_radar_scan")
    if not scan:
        st.info("Run Market Radar and it will replace the giant odds list with a ranked board of only the unusual stuff.")
        return

    offers = scan.get("offers")
    board = scan.get("board")
    # Re-apply the live watchlist without spending API credits again.
    board = apply_watchlist(board, watch_terms) if isinstance(board, pd.DataFrame) else pd.DataFrame()
    scanned_at = scan.get("scanned_at")
    try:
        scan_et = pd.Timestamp(scanned_at).tz_convert("America/New_York").strftime("%b %-d, %-I:%M:%S %p ET")
    except Exception:
        scan_et = str(scanned_at or "")
    st.caption(f"Last radar scan: **{scan_et}** · {scan.get('scan_mode')} · {scan.get('horizon')} horizon. Cached scans do not spend credits again until the API cache expires or inputs change.")

    if board.empty:
        st.info("No opportunities cleared the current sensitivity thresholds. That can be a good sign: the selected books are closely aligned right now.")
    else:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Flagged", len(board))
        m2.metric("True arbs", int(board["Type"].astype(str).str.startswith("TRUE ARB").sum()) + int((board["Type"] == "ARB + MIDDLE").sum()))
        m3.metric("Middles", int(board["Type"].isin(["MIDDLE", "ARB + MIDDLE"]).sum()))
        m4.metric("Line outliers", int((board["Type"] == "LINE OUTLIER").sum()))
        m5.metric("Watchlist hits", int(board.get("Watch", pd.Series(dtype=bool)).sum()))

        mode_col, group_col = st.columns([1.2, 1.8])
        board_mode = mode_col.selectbox(
            "Board view",
            ["Balanced", "Arbs & middles", "Game lines", "Core player props", "TD / scorer props", "All signals"],
            index=0, key="radar_board_view",
            help="Balanced is recommended. All signals shows the raw ranked output with no diversity cap."
        )
        available_groups = sorted(board.get("Market Group", pd.Series(["Other markets"])).fillna("Other markets").astype(str).unique())
        filter_groups = group_col.multiselect("Market groups", available_groups, default=available_groups, key="radar_market_groups")

        f1, f2, f3, f4 = st.columns([1.3, 1.2, 1.2, 1.0])
        all_types = list(dict.fromkeys(board["Type"].astype(str).tolist()))
        filter_types = f1.multiselect("Opportunity types", all_types, default=all_types, key="radar_filter_types")
        filter_sports = f2.multiselect("Sports", sorted(board["Sport"].astype(str).unique()), default=sorted(board["Sport"].astype(str).unique()), key="radar_filter_sports")
        min_score = f3.slider("Minimum mispricing score", 0, 100, 45, 1, key="radar_min_score")
        watch_only = f4.checkbox("Watchlist only", value=False, key="radar_watch_only")
        search_text = st.text_input("Search player / team / market", value="", key="radar_search", placeholder="e.g. Josh Allen, Steelers, receiving yards")

        view = board[
            board["Type"].isin(filter_types)
            & board["Sport"].astype(str).isin(filter_sports)
            & board.get("Market Group", pd.Series("Other markets", index=board.index)).astype(str).isin(filter_groups)
            & (pd.to_numeric(board["Score"], errors="coerce") >= float(min_score))
        ].copy()
        if board_mode == "Arbs & middles":
            view = view[view["Type"].astype(str).str.startswith("TRUE ARB") | view["Type"].astype(str).isin(["ARB + MIDDLE", "MIDDLE"])]
        elif board_mode == "Game lines":
            view = view[view.get("Market Group", "").astype(str).eq("Game lines")]
        elif board_mode == "Core player props":
            view = view[view.get("Market Group", "").astype(str).eq("Core player props")]
        elif board_mode == "TD / scorer props":
            view = view[view.get("Market Group", "").astype(str).eq("TD / scorer props")]
        elif board_mode == "Balanced":
            view = _balanced_radar_view(view, max_rows=100)
        if watch_only and "Watch" in view.columns:
            view = view[view["Watch"]]
        if search_text.strip():
            q = search_text.strip().lower()
            blob_cols = ["Sport", "Matchup", "Market", "Subject", "Pick", "Book", "Book 2", "Why"]
            mask = view[blob_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower().str.contains(q, regex=False)
            view = view[mask]

        display = view.head(100).copy()
        if display.empty:
            st.caption("Nothing matches the current filters.")
        else:
            display.insert(0, "⭐", display.get("Watch", False).map(lambda x: "⭐" if bool(x) else ""))
            display["Signal"] = display["Type"].map(_radar_type_icon)
            display["Odds"] = display["Odds"].map(lambda x: fmt_odds(x) if pd.notna(x) else "—")
            display["Odds 2"] = display["Odds 2"].map(lambda x: fmt_odds(x) if pd.notna(x) else "")
            cols = ["⭐", "Score", "Signal", "Sport", "Market Group", "Matchup", "Market", "Pick", "Book", "Odds", "Book 2", "Odds 2", "Edge", "Compare"]
            st.dataframe(
                display[cols], width="stretch", hide_index=True,
                column_config={
                    "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
                    "Signal": st.column_config.TextColumn("Opportunity"),
                },
            )
            st.download_button(
                "Download current opportunity board CSV",
                data=view.to_csv(index=False).encode("utf-8"),
                file_name="market_radar_opportunities.csv",
                mime="text/csv",
                key="radar_download",
            )

            # Inspector: show why it was flagged and every underlying offer for that market.
            labels = [f"{i+1}. {_radar_type_icon(r['Type'])} — {r['Sport']} — {r['Pick']} — {r['Edge']}" for i, (_, r) in enumerate(view.head(100).iterrows())]
            selected_label = st.selectbox("Inspect an opportunity", labels, index=0, key="radar_inspect")
            pos = labels.index(selected_label)
            rr = view.head(100).iloc[pos]
            box = st.container(border=True)
            with box:
                h1, h2, h3 = st.columns([1.2, 1.2, 1.0])
                h1.metric("Mispricing score", f"{float(rr['Score']):.0f}/100")
                h2.metric("Edge", str(rr.get("Edge") or "—"))
                h3.metric("Type", str(rr.get("Type") or "—"))
                st.markdown(f"**{rr.get('Sport')} · {rr.get('Matchup')} · {rr.get('Market')}**")
                st.write(rr.get("Why") or "")
                if str(rr.get("Details") or "").strip():
                    st.info(str(rr.get("Details")))
                if bool(rr.get("Watch")):
                    st.success("⭐ This matches one of the sides/players you said you already like.")

                if isinstance(offers, pd.DataFrame) and not offers.empty:
                    source = offers[
                        offers["Event ID"].astype(str).eq(str(rr.get("Event ID")))
                        & offers["Market Key"].astype(str).eq(str(rr.get("Market Key")))
                    ].copy()
                    subject = str(rr.get("Subject") or "")
                    if subject and subject not in {"Game", "Championship"}:
                        narrower = source[source["Subject"].astype(str).eq(subject)]
                        if not narrower.empty:
                            source = narrower
                    if not source.empty:
                        source["Quote age"] = source["Updated"].map(_quote_age_text)
                        source["Odds"] = source["Odds"].map(fmt_odds)
                        source["Line"] = source["Line"].map(lambda x: "—" if pd.isna(x) else f"{float(x):g}")
                        raw_cols = ["Book", "Side", "Line", "Odds", "Quote age", "Updated", "Bet Limit", "Link"]
                        st.markdown("**Underlying prices used by the scanner**")
                        st.dataframe(
                            source[raw_cols].sort_values(["Side", "Line", "Book"], na_position="last"),
                            width="stretch", hide_index=True,
                            column_config={"Link": st.column_config.LinkColumn("Open book", display_text="Open")},
                        )

                # One-market uncached verification. This is deliberately separate from the
                # broad scan so the user does not need to spend credits rescanning every game.
                event_id = str(rr.get("Event ID") or "")
                market_key = str(rr.get("Market Key") or "")
                league = str(rr.get("Sport") or "")
                can_live_check = bool(event_id and market_key and market_key != "outrights" and league in SPORT_KEYS)
                if can_live_check:
                    if st.button("🔄 Verify this market live", key=f"radar_live_{event_id}_{market_key}_{pos}"):
                        try:
                            live_payload, live_usage = fetch_sport_event_odds_live(
                                api_key, SPORT_KEYS[league], event_id, market_key, bookmaker_csv
                            )
                            remember_odds_api_usage(live_usage)
                            live_frame = normalize_market_payloads(live_payload, league)
                            st.session_state["radar_live_verify"] = {
                                "event_id": event_id, "market_key": market_key,
                                "league": league, "frame": live_frame,
                                "checked_at": datetime.now(timezone.utc).isoformat(),
                            }
                        except Exception as exc:
                            st.error(f"Live verification failed: {exc}")

                    live_state = st.session_state.get("radar_live_verify") or {}
                    if live_state.get("event_id") == event_id and live_state.get("market_key") == market_key:
                        lf = live_state.get("frame")
                        if isinstance(lf, pd.DataFrame) and not lf.empty:
                            subject = str(rr.get("Subject") or "")
                            if subject and subject not in {"Game", "Championship"}:
                                n = lf[lf["Subject"].astype(str).eq(subject)].copy()
                                if not n.empty:
                                    lf = n
                            lf = lf.copy()
                            lf["Quote age"] = lf["Updated"].map(_quote_age_text)
                            lf["Odds"] = lf["Odds"].map(fmt_odds)
                            lf["Line"] = lf["Line"].map(lambda x: "—" if pd.isna(x) else f"{float(x):g}")
                            checked = live_state.get("checked_at")
                            try:
                                checked = pd.Timestamp(checked).tz_convert("America/New_York").strftime("%-I:%M:%S %p ET")
                            except Exception:
                                pass
                            st.success(f"Fresh one-market verification completed at {checked}.")
                            st.dataframe(
                                lf[["Book", "Side", "Line", "Odds", "Quote age", "Updated", "Bet Limit", "Link"]]
                                  .sort_values(["Side", "Line", "Book"], na_position="last"),
                                width="stretch", hide_index=True,
                                column_config={"Link": st.column_config.LinkColumn("Open book", display_text="Open")},
                            )
                        else:
                            st.warning("The fresh request returned no offers for this market.")
                    st.caption("Live verification requests only this one market for this one event. It can use API credits, but avoids rerunning the full radar.")

        with st.expander("Book / exchange coverage from this scan"):
            cov = book_coverage_summary(offers, scan.get("book_keys")) if isinstance(offers, pd.DataFrame) else pd.DataFrame()
            if cov.empty:
                st.caption("No bookmaker offers were returned.")
            else:
                st.dataframe(cov, width="stretch", hide_index=True)

    if scan.get("errors"):
        with st.expander(f"Scan warnings ({len(scan['errors'])})"):
            for x in scan["errors"]:
                st.write("•", x)

    st.caption(
        "Score is an anomaly-ranking score, not a win probability. Arb math uses the exchange-fee safety buffer you selected; always verify the live executable price, limits, fees and liquidity before betting. "
        "Middles can lose. Price/line outliers simply mean one of your books differs materially from the others—useful when it happens to be on a side you already like."
    )


# -----------------------------
# UI
# -----------------------------
st.title("🏈 NFL Betting Lab + Market Radar")
st.caption("NFL model lab + automatic NFL/NBA/MLB/NHL market anomaly scanning across the books and exchanges you actually use.")


with st.sidebar:
    st.header("Live data settings")
    secret_key = ""
    try:
        secret_key = str(st.secrets.get("THE_ODDS_API_KEY", "") or "")
    except Exception:
        secret_key = ""

    entered_key = st.text_input(
        "The Odds API key",
        value="",
        type="password",
        help="Optional. Put the key in Streamlit Secrets for a deployed app, or paste it here for this session.",
    )
    api_key = entered_key.strip() or secret_key.strip()

    selected_book_labels = st.multiselect(
        "My sportsbooks & exchanges",
        options=list(BOOKMAKER_OPTIONS.keys()),
        default=DEFAULT_BOOKS,
        help="Used by Market Radar and the NFL line-shop tools. Up to 10 selected bookmaker keys are billed like one region by The Odds API. Caesars requires a paid Odds API subscription.",
    )
    selected_book_keys = [BOOKMAKER_OPTIONS[x] for x in selected_book_labels]
    bookmaker_csv = ",".join(selected_book_keys)
    if len(selected_book_keys) > 10:
        st.caption("More than 10 books increases The Odds API credit cost because each group of 10 counts like another region.")

    odds_credit_panel = st.container(border=True)

    strict_mode = st.checkbox(
        "Strict V1 ratings",
        value=False,
        help="Strict mode requires at least four prior games in the CURRENT season. Turn it off to use an experimental prior-season carryover early in the year.",
    )
    st.caption("Odds key is never written into the app files.")

    if st.button("Refresh live NFL data", help="Clears cached schedule/PBP/odds/injury/referee data and reloads the latest public feeds."):
        st.cache_data.clear()
        st.rerun()

try:
    with st.spinner("Loading NFL schedule…"):
        schedule = load_schedule()
except Exception as exc:
    st.exception(exc)
    st.stop()

render_forward_test_panel(schedule)

# Pikkit-style multi-sport anomaly detector lives near the top of the app so it can be used
# without scrolling through the NFL model sections first.
render_market_radar(api_key, selected_book_keys, bookmaker_csv)

upcoming = upcoming_schedule(schedule, days_ahead=12)
if upcoming.empty:
    st.warning("No regular-season games were found in the next 12 days.")
    st.stop()

# Friendly game labels.
game_options = []
for idx, r in upcoming.iterrows():
    label = f"{pd.Timestamp(r['gameday']).strftime('%a %b %d')} — {r['away_team']} @ {r['home_team']} — Week {int(r['week'])}"
    game_options.append((idx, label))

chosen_label = st.selectbox("Choose game", [x[1] for x in game_options])
chosen_idx = next(x[0] for x in game_options if x[1] == chosen_label)
game = upcoming.loc[chosen_idx]

away = str(game["away_team"])
home = str(game["home_team"])
week = int(game["week"])
season = int(game["season"])
kickoff_utc = schedule_kickoff_utc(game)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Away", away)
c2.metric("Home", home)
c3.metric("Week", week)
c4.metric("Kickoff (ET)", kickoff_utc.astimezone(ZoneInfo("America/New_York")).strftime("%a %-I:%M %p"))

# -----------------------------
# Odds
# -----------------------------
odds_events = []
event = None
book_rows = []
odds_error = None
if api_key:
    try:
        with st.spinner("Refreshing sportsbook odds…"):
            odds_events, odds_usage = fetch_odds(api_key, bookmaker_csv)
            remember_odds_api_usage(odds_usage)
        event = match_odds_event(odds_events, away, home)
        book_rows = book_market_rows(event)
    except Exception as exc:
        odds_error = str(exc)

if odds_error:
    st.warning(f"Live odds unavailable: {odds_error}")

with odds_credit_panel:
    render_odds_api_credit_panel()

st.subheader("Market")
market_source = "Manual / nflverse snapshot"
selected_book_row = None

if book_rows:
    books_df = pd.DataFrame(book_rows)
    default_index = 0
    dk = books_df.index[books_df["key"].eq("draftkings")].tolist()
    if dk:
        default_index = dk[0]
    book_name = st.selectbox("Sportsbook", books_df["book"].tolist(), index=default_index)
    selected_book_row = books_df[books_df["book"] == book_name].iloc[0].to_dict()
    market_source = book_name

    with st.expander("Compare available sportsbook totals"):
        show_cols = [
            "book", "total", "over", "under",
            "home_spread", "home_spread_odds", "away_spread_odds",
            "home_ml", "away_ml", "updated"
        ]
        show = books_df[[c for c in show_cols if c in books_df.columns]].copy()
        st.dataframe(show, use_container_width=True, hide_index=True)

fallback_total = to_float(game.get("total_line"), 44.5)
fallback_over = to_float(game.get("over_odds"), -110)
fallback_under = to_float(game.get("under_odds"), -110)
fallback_spread = -to_float(game.get("spread_line"), 0.0)

if selected_book_row:
    default_total = selected_book_row["total"] or fallback_total
    default_over = selected_book_row["over"] or fallback_over
    default_under = selected_book_row["under"] or fallback_under
    default_spread = selected_book_row["home_spread"] if selected_book_row["home_spread"] is not None else fallback_spread
else:
    default_total, default_over, default_under, default_spread = fallback_total, fallback_over, fallback_under, fallback_spread

m1, m2, m3, m4 = st.columns(4)
market_total = m1.number_input("Total", value=float(default_total), step=0.5)
over_odds = m2.number_input("Over odds", value=int(round(default_over)), step=1)
under_odds = m3.number_input("Under odds", value=int(round(default_under)), step=1)
home_spread = m4.number_input("Home spread", value=float(default_spread), step=0.5)
st.caption(f"Market source: **{market_source}**. You can always overwrite the line or price before calculating.")

# Current spread and moneyline prices from the selected sportsbook.
if selected_book_row:
    default_home_spread_odds = selected_book_row.get("home_spread_odds")
    default_away_spread_odds = selected_book_row.get("away_spread_odds")
    default_home_ml = selected_book_row.get("home_ml")
    default_away_ml = selected_book_row.get("away_ml")
else:
    default_home_spread_odds = None
    default_away_spread_odds = None
    default_home_ml = None
    default_away_ml = None

st.markdown("##### Spread & moneyline prices")
s1, s2, s3 = st.columns(3)
home_spread_odds = s1.number_input(
    f"{home} {home_spread:+.1f} odds",
    value=int(round(default_home_spread_odds if default_home_spread_odds is not None else -110)),
    step=1,
)
away_spread_odds = s2.number_input(
    f"{away} {-home_spread:+.1f} odds",
    value=int(round(default_away_spread_odds if default_away_spread_odds is not None else -110)),
    step=1,
)
home_ml = s3.number_input(
    f"{home} moneyline",
    value=int(round(default_home_ml if default_home_ml is not None else -110)),
    step=1,
)

s4, s5, s6 = st.columns(3)
away_ml = s4.number_input(
    f"{away} moneyline",
    value=int(round(default_away_ml if default_away_ml is not None else -110)),
    step=1,
)
s5.metric("Away spread", f"{-home_spread:+.1f}")
s6.caption("These prices update from the selected sportsbook when available.")


# -----------------------------
# Weather + roof
# -----------------------------
st.subheader("Weather")
weather = None
weather_error = None
weather_loc = WEATHER_COORDS.get(home)

try:
    if weather_loc:
        lat, lon, venue_name = weather_loc
        with st.spinner("Loading stadium-area forecast…"):
            weather = fetch_weather(lat, lon, venue_name, kickoff_utc.isoformat())
except Exception as exc:
    weather_error = str(exc)

roof_type = "fixed" if home in FIXED_ROOF else "retractable" if home in RETRACTABLE_ROOF else "outdoor"
schedule_roof = str(game.get("roof") or "").lower()

if roof_type == "fixed":
    dome = 1.0
    roof_label = "Fixed roof / covered"
elif roof_type == "outdoor":
    dome = 0.0
    roof_label = "Outdoor"
else:
    if schedule_roof in {"closed", "dome"}:
        default_roof = "Closed"
    elif schedule_roof == "open":
        default_roof = "Open"
    else:
        default_roof = "Unknown / assume open"
    roof_choice = st.selectbox(
        "Retractable roof status",
        ["Unknown / assume open", "Open", "Closed"],
        index=["Unknown / assume open", "Open", "Closed"].index(default_roof),
        help="Retractable-roof decisions may not be known until game day. Confirm this before relying on the weather input.",
    )
    dome = 1.0 if roof_choice == "Closed" else 0.0
    roof_label = f"Retractable — {roof_choice}"

if weather:
    outside_temp = weather["temperature"]
    outside_wind = weather["wind"]
    outside_gust = weather["gust"]
    precip = weather["precip"]
    if dome == 1:
        model_temp = 70.0
        model_wind = 0.0
    else:
        model_temp = outside_temp
        model_wind = outside_wind

    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Roof", roof_label)
    w2.metric("Kickoff temp", f"{outside_temp:.0f}°F")
    w3.metric("Wind", f"{outside_wind:.0f} mph")
    w4.metric("Rain chance", f"{precip:.0f}%")
    st.caption(
        f"Forecast location: {weather['resolved_name']} · local kickoff "
        f"{weather['kickoff_local'].strftime('%a %-I:%M %p')} · gusts {outside_gust:.0f} mph. "
        + ("Indoor model inputs use 70°F / 0 mph." if dome == 1 else "Outdoor model inputs use the forecast values.")
    )
else:
    if weather_error:
        st.warning(f"Weather could not load: {weather_error}")
    model_temp = 70.0
    model_wind = 0.0 if dome == 1 else st.number_input("Manual wind mph", value=5.0)
    if dome == 0:
        model_temp = st.number_input("Manual temperature °F", value=70.0)

cold_units = 0.0 if dome == 1 else max(0.0, (40.0 - model_temp) / 10.0)

# -----------------------------
# Team ratings
# -----------------------------
st.subheader("Automatic team ratings")

try:
    with st.spinner(f"Loading {season} nflverse play-by-play…"):
        cur_pbp = load_pbp(season)
    with st.spinner(f"Loading {season-1} prior-season play-by-play…"):
        prev_pbp = load_pbp(season - 1)

    # Freshness audit: Week 2+ predictions should consume every available game
    # from earlier weeks, but never the selected/current week's outcomes.
    pbp_week = pd.to_numeric(cur_pbp.get("week"), errors="coerce")
    prior_mask = pbp_week.notna() & (pbp_week > 0) & (pbp_week < week)
    prior_pbp = cur_pbp.loc[prior_mask].copy()
    prior_teams = set(prior_pbp.get("posteam", pd.Series(dtype=str)).dropna().astype(str)) | set(prior_pbp.get("defteam", pd.Series(dtype=str)).dropna().astype(str))
    latest_prior_week = int(pbp_week[prior_mask].max()) if prior_mask.any() else 0
    if week >= 2:
        if latest_prior_week >= week - 1 and len(prior_teams) >= 32:
            st.success(f"Live-data refresh: Week {week-1} PBP is loaded for all 32 teams. Week {week} features now include those results.")
        elif latest_prior_week >= 1:
            st.warning(f"Live-data refresh: PBP is available through Week {latest_prior_week} for {len(prior_teams)}/32 teams. The remaining team feeds may still be waiting on a late game/provider update.")
        else:
            st.warning("Live-data refresh: current-season prior-week PBP has not populated yet. Use ‘Refresh live NFL data’ later before trusting Week 2+ estimates.")

    cur_metrics = aggregate_game_metrics(cur_pbp)
    prev_metrics = aggregate_game_metrics(prev_pbp)

    cur_states = build_histories(cur_metrics, max_week=week)
    prev_states = build_histories(prev_metrics, max_week=None)

    football_features, rating_meta = matchup_features(
        away, home, prev_states, cur_states, strict=strict_mode
    )
except Exception as exc:
    st.error(f"Could not build nflverse team ratings: {exc}")
    football_features = None
    rating_meta = {"away_games": 0, "home_games": 0, "used_prior": False}

r1, r2, r3 = st.columns(3)
r1.metric(f"{away} current-season games used", rating_meta["away_games"])
r2.metric(f"{home} current-season games used", rating_meta["home_games"])
r3.metric("Prior-season carryover", "Yes" if rating_meta["used_prior"] else "No")

if rating_meta["used_prior"]:
    st.warning(
        "Early-season mode is using prior-season games because one or both teams have fewer than four "
        "current-season games. This makes the output **experimental** and is not the same setup that was backtested."
    )

if football_features is None:
    st.warning(
        "Not enough pregame team data under the selected rating mode. Turn off Strict V1 early in the season "
        "to see the experimental prior-season carryover."
    )
    st.stop()

with st.expander("See the advanced inputs the app calculated"):
    feature_table = pd.DataFrame(
        [{"Feature": k, "Value": v} for k, v in football_features.items()]
    )
    st.dataframe(feature_table, use_container_width=True, hide_index=True)

# Rest/division inputs from schedule.
away_rest = to_float(game.get("away_rest"), 7.0)
home_rest = to_float(game.get("home_rest"), 7.0)
short_rest = 1.0 if min(away_rest, home_rest) <= 5 else 0.0
div_raw = game.get("div_game")
try:
    div_game = 1.0 if float(div_raw) == 1 else 0.0
except Exception:
    div_game = 0.0

all_features = dict(football_features)
all_features.update(
    {
        "Wind": float(model_wind),
        "Cold_Units": float(cold_units),
        "Dome": float(dome),
        "Short_Rest": short_rest,
        "Div_Game": div_game,
    }
)



# -----------------------------
# Weekly model favorites board
# -----------------------------
st.markdown("---")
st.subheader("Weekly favorites scanner")
st.caption("Builds the strongest three ML, spread and total opinions for the selected NFL week. This is a confidence ranking, not an EV ranking.")
week_games = upcoming[(pd.to_numeric(upcoming["season"], errors="coerce") == season) & (pd.to_numeric(upcoming["week"], errors="coerce") == week)].copy()
board_key = f"weekly_favorites_{season}_{week}"
if st.button(f"Build / refresh Week {week} model favorites", type="primary", key=f"weekly_favorites_button_{season}_{week}"):
    if not api_key:
        st.warning("Add The Odds API key first so the weekly board can use current moneyline/spread/total markets.")
    elif build_side_prediction_v25 is None and build_side_prediction_v24 is None:
        st.warning(f"Side models are unavailable. V2.5: {V25_SIDE_IMPORT_ERROR}; V2.4: {V24_SIDE_IMPORT_ERROR}")
    else:
        try:
            side_label = "V2.5" if build_side_prediction_v25 is not None else "V2.4"
            with st.spinner(f"Scanning Week {week} games with {side_label} sides + V1 totals…"):
                st.session_state[board_key] = build_weekly_favorites_board(
                    schedule=schedule, week_games=week_games, odds_events=odds_events,
                    prev_pbp=prev_pbp, cur_pbp=cur_pbp, prev_states=prev_states, cur_states=cur_states,
                    season=season, week=week, strict_mode=strict_mode, root=ROOT,
                )
        except Exception as exc:
            st.error(f"Weekly favorites scan failed: {exc}")

if board_key in st.session_state:
    render_weekly_favorites(st.session_state[board_key], week)
else:
    st.info("Click the button above after odds have loaded. The scan checks every game in the selected week and keeps the top three in each market.")

st.markdown("---")
st.subheader("💰 NFL detailed line shop, player props & arbitrage")
st.caption(
    "Player-prop EV here is a **de-vigged market-consensus estimate**, not a trained player-prop prediction model yet. "
    "The scanner compares the best offered price with other sportsbooks at the exact same player/market/line. "
    "Arbitrage is math-only and does not depend on the NFL prediction model."
)

if not api_key:
    st.info("Add The Odds API key to use multi-book player props and arbitrage scanning.")
else:
    tab_lines, tab_props, tab_arb = st.tabs(["Best lines", "Player prop value", "Arbitrage & middles"])

    with tab_lines:
        best_lines = core_best_lines(odds_events)
        if best_lines.empty:
            st.caption("No multi-book core markets are available from the selected sportsbooks right now.")
        else:
            selected_matchup_name = f"{ABBR_TO_ODDS_NAME.get(away, away)} @ {ABBR_TO_ODDS_NAME.get(home, home)}"
            one = best_lines[best_lines["Matchup"].eq(selected_matchup_name)].copy()
            if one.empty:
                one = best_lines.copy()
            if "Odds" in one:
                one["Odds"] = one["Odds"].map(fmt_odds)
            if "Line" in one:
                one["Line"] = one["Line"].map(lambda x: "—" if pd.isna(x) else f"{float(x):+.1f}")
            st.dataframe(one, width="stretch", hide_index=True)
            st.caption("For spreads, the finder prioritizes the most favorable point, then the best price. For totals, Over prefers the lowest line and Under the highest line.")

    with tab_props:
        scope = st.radio("Prop scan scope", ["Selected game", f"Week {week}"], horizontal=True, key="prop_scope")
        prop_label_to_key = {v: k for k, v in PROP_MARKET_LABELS.items() if k != "player_anytime_td"}
        default_prop_labels = [PROP_MARKET_LABELS[k] for k in CORE_PROP_MARKETS if k in PROP_MARKET_LABELS]
        chosen_prop_labels = st.multiselect(
            "Player prop markets",
            options=list(prop_label_to_key.keys()),
            default=default_prop_labels,
            key="prop_markets",
        )
        chosen_prop_keys = [prop_label_to_key[x] for x in chosen_prop_labels]
        p1, p2, p3 = st.columns(3)
        min_prop_ev = p1.number_input("Minimum displayed EV %", value=2.0, step=0.5, key="min_prop_ev") / 100.0
        min_other_books = p2.selectbox("Minimum other consensus books", [1, 2, 3], index=0, key="prop_consensus_n")
        max_prop_rows = p3.selectbox("Show top", [10, 20, 30, 50], index=1, key="prop_top_n")

        # Find the Odds API event IDs without spending quota (events endpoint is free).
        prop_scan_key = f"prop_scan_{season}_{week}_{scope}_{','.join(chosen_prop_keys)}_{bookmaker_csv}"
        target_events = []
        try:
            nfl_events, events_usage = fetch_nfl_events(api_key)
            remember_odds_api_usage(events_usage)
            if scope == "Selected game":
                ev = match_odds_event(nfl_events, away, home)
                if ev:
                    target_events = [ev]
            else:
                for _, wg in week_games.iterrows():
                    ev = match_odds_event(nfl_events, str(wg["away_team"]), str(wg["home_team"]))
                    if ev:
                        target_events.append(ev)
        except Exception as exc:
            st.warning(f"Could not load event IDs: {exc}")

        regions_equiv = max(1, math.ceil(max(1, len(selected_book_keys)) / 10))
        estimated_credits = len(target_events) * len(chosen_prop_keys) * regions_equiv
        usage_now = current_odds_api_usage()
        credits_remaining = usage_now.get("remaining")
        remaining_after = None if credits_remaining is None else int(credits_remaining) - int(estimated_credits)

        st.caption(
            f"Maximum estimated prop-scan cost: about **{estimated_credits} API credits** "
            f"({len(target_events)} event(s) × {len(chosen_prop_keys)} markets × {regions_equiv} bookmaker group(s)). "
            "Actual cost can be lower when requested markets are unavailable."
        )
        if credits_remaining is not None:
            st.caption(
                f"Current quota: **{int(credits_remaining):,} credits remaining** → "
                f"approximately **{max(0, remaining_after):,} remaining after this scan** at the maximum estimated cost."
            )
            if estimated_credits > int(credits_remaining):
                st.error("This scan's estimated maximum cost is greater than your remaining quota. Reduce games, markets or sportsbooks first.")
            elif remaining_after < 50 or estimated_credits >= max(25, int(credits_remaining) * 0.25):
                st.warning("This is a relatively expensive scan for your remaining quota. Consider fewer prop markets or the selected-game scope.")

        scan_disabled = bool(
            not chosen_prop_keys
            or not target_events
            or (credits_remaining is not None and estimated_credits > int(credits_remaining))
        )
        if st.button("Scan player props", type="primary", key="scan_props", disabled=scan_disabled):
            payloads = []
            errors = []
            progress = st.progress(0.0)
            for i, ev in enumerate(target_events, start=1):
                try:
                    x, prop_usage = fetch_event_props(api_key, str(ev["id"]), ",".join(chosen_prop_keys), bookmaker_csv)
                    remember_odds_api_usage(prop_usage)
                    if x:
                        payloads.append(x)
                except Exception as exc:
                    errors.append(f"{ev.get('away_team')} @ {ev.get('home_team')}: {exc}")
                progress.progress(i / max(1, len(target_events)))
            progress.empty()
            st.session_state[prop_scan_key] = {"payloads": payloads, "errors": errors}
            # Keep latest scan available to the arb tab without another API call.
            st.session_state["latest_prop_payloads"] = payloads
            # Rerun once so the sidebar immediately reflects the newest quota counters.
            st.rerun()

        scan_state = st.session_state.get(prop_scan_key)
        if scan_state:
            prop_values = prop_value_table(scan_state.get("payloads", []), minimum_other_books=min_other_books)
            if prop_values.empty:
                st.warning("No exact-line player props had enough other-book consensus to calculate fair EV.")
            else:
                prop_values = prop_values[prop_values["EV"] >= min_prop_ev].copy().head(int(max_prop_rows))
                if prop_values.empty:
                    st.info("No props cleared the selected EV threshold.")
                else:
                    show = prop_values[[
                        "Matchup", "Player", "Market", "Side", "Line", "Odds", "Book",
                        "Fair Prob", "Fair Odds", "EV", "Probability Edge", "Consensus Books"
                    ]].copy()
                    show["Odds"] = show["Odds"].map(fmt_odds)
                    show["Fair Odds"] = show["Fair Odds"].map(fmt_odds)
                    show["Fair Prob"] = show["Fair Prob"].map(lambda x: f"{float(x):.1%}")
                    show["EV"] = show["EV"].map(lambda x: f"{float(x):+.1%}")
                    show["Probability Edge"] = show["Probability Edge"].map(lambda x: f"{float(x):+.1%}")
                    show["Line"] = show["Line"].map(lambda x: "—" if pd.isna(x) else f"{float(x):g}")
                    st.dataframe(show, width="stretch", hide_index=True)
                    st.warning("These are market-consensus value estimates, not independently backtested player projections. Verify the line is still available before betting.")
            if scan_state.get("errors"):
                with st.expander(f"Prop scan errors ({len(scan_state['errors'])})"):
                    for e in scan_state["errors"]:
                        st.write("•", e)
        else:
            st.info("Choose the scope and markets, then click **Scan player props**. Scans are cached for about 90 seconds to reduce API usage.")

    with tab_arb:
        st.caption("Legacy NFL exact-line view. The multi-sport Market Radar above is the preferred scanner because it can apply the configurable exchange-fee buffer and rank mispriced lines, not just exact arbs.")
        if any(k in {"novig", "kalshi", "polymarket", "prophetx"} for k in selected_book_keys):
            st.warning("This legacy tab uses raw displayed exchange prices and does not model exchange fees. Treat exchange-involved rows as candidates to verify, not locked profit. Market Radar is fee-buffer aware.")
        a1, a2 = st.columns(2)
        arb_stake = a1.number_input("Total stake for arb calculator ($)", min_value=1.0, value=100.0, step=25.0, key="arb_stake")
        min_arb_roi = a2.number_input("Minimum raw-price ROI %", min_value=0.0, value=0.0, step=0.1, key="arb_min_roi") / 100.0
        core_arbs = core_arbitrage_table(odds_events, total_stake=arb_stake, min_roi=min_arb_roi)
        st.markdown("#### Potential exact arbitrage — moneylines, spreads & totals")
        if core_arbs.empty:
            st.info("No true core-market arbitrage is visible across the selected books right now.")
        else:
            z = core_arbs.copy()
            for c in ["Odds A", "Odds B"]:
                z[c] = z[c].map(fmt_odds)
            z["ROI"] = z["ROI"].map(lambda x: f"{float(x):.2%}")
            for c in ["Stake A", "Stake B", "Guaranteed Payout", "Guaranteed Profit"]:
                z[c] = z[c].map(lambda x: f"${float(x):,.2f}")
            st.dataframe(z, width="stretch", hide_index=True)

        payloads = st.session_state.get("latest_prop_payloads", [])
        st.markdown("#### Potential exact arbitrage — player props")
        if payloads:
            parbs = prop_arbitrage_table(payloads, total_stake=arb_stake, min_roi=min_arb_roi)
            if parbs.empty:
                st.info("No true player-prop arbitrage was found in your latest prop scan.")
            else:
                z = parbs.copy()
                for c in ["Odds A", "Odds B"]:
                    z[c] = z[c].map(fmt_odds)
                z["ROI"] = z["ROI"].map(lambda x: f"{float(x):.2%}")
                for c in ["Stake A", "Stake B", "Guaranteed Payout", "Guaranteed Profit"]:
                    z[c] = z[c].map(lambda x: f"${float(x):,.2f}")
                st.dataframe(z, width="stretch", hide_index=True)
        else:
            st.caption("Run the Player Prop Value scan first; the arb scanner reuses that data so it does not spend API credits twice.")

        st.markdown("#### Middles (not guaranteed arbitrage)")
        middles = core_middle_table(odds_events)
        if middles.empty:
            st.caption("No useful spread/total line middles are visible across the selected books.")
        else:
            st.dataframe(middles, width="stretch", hide_index=True)
        st.warning("Arbitrage opportunities can disappear within seconds, and books can limit/void stale or erroneous prices. Always verify both legs before placing either bet. Middles can lose and are not guaranteed-profit bets.")

st.markdown("---")

# -----------------------------
# Model result
# -----------------------------
st.subheader("Model output")
result = model_output(
    all_features,
    float(market_total),
    float(over_odds),
    float(under_odds),
)

a, b, c, d = st.columns(4)
a.metric("Sportsbook total", f"{market_total:.1f}")
b.metric("V1 model total", f"{result['model_total']:.1f}")
c.metric("Model edge", f"{result['correction']:+.2f}")
d.metric("V1 signal", result["signal"])

a, b, c, d = st.columns(4)
a.metric("Over probability", f"{result['p_over']:.1%}")
b.metric("Under probability", f"{result['p_under']:.1%}")
c.metric("Fair Over", fmt_odds(result["fair_over"]))
d.metric("Fair Under", fmt_odds(result["fair_under"]))

a, b, c = st.columns(3)
a.metric("Over EV", f"{result['over_ev']:+.1%}")
b.metric("Under EV", f"{result['under_ev']:+.1%}")
c.metric("Weather input", f"{model_temp:.0f}°F / {model_wind:.0f} mph")

if result["signal"] != "PASS":
    st.warning(
        f"V1 totals signal: **{result['signal']}**. "
        "V1 lost 18.2% ROI in its 2025 out-of-sample backtest, so the totals model remains experimental."
    )
else:
    st.info("V1 filter says PASS.")

# -----------------------------
# V2.5 Challenger — LightGBM residual ensemble
# -----------------------------
st.subheader("V2.5 Challenger — Best Current Side Model")
st.caption(
    "V2.5 is a new 2026 challenger trained only on 2018–2025 games. It uses the same live-generatable feature families as the existing side pipeline, "
    "but replaces the older Ridge/HistGradientBoosting margin engine with a two-model LightGBM residual ensemble and separately calibrated ML/spread probabilities. "
    "All 2021–2025 results were used for development; 2026 remains the clean forward test."
)

v25_result = None
v25_error = None
if build_side_prediction_v25 is None:
    v25_error = f"V2.5 module could not load: {V25_SIDE_IMPORT_ERROR}"
else:
    try:
        with st.spinner("Running V2.5 LightGBM challenger…"):
            v25_result = build_side_prediction_v25(
                root=ROOT,
                schedule=schedule,
                game=game,
                prev_pbp=prev_pbp,
                cur_pbp=cur_pbp,
                home=home,
                away=away,
                season=season,
                week=week,
                market_total=float(market_total),
                over_odds=float(over_odds),
                under_odds=float(under_odds),
                home_spread=float(home_spread),
                home_spread_odds=float(home_spread_odds),
                away_spread_odds=float(away_spread_odds),
                home_ml=float(home_ml),
                away_ml=float(away_ml),
                dome=float(dome),
            )
    except Exception as exc:
        v25_error = str(exc)

if v25_error:
    st.error(f"V2.5 challenger could not run: {v25_error}")

if v25_result:
    margin = float(v25_result["predicted_margin"])
    margin_label = f"{home} by {margin:.1f}" if margin >= 0 else f"{away} by {abs(margin):.1f}"
    market_home_margin = -float(home_spread)
    edge_pts = margin - market_home_margin
    eligible_cov = float(v25_result.get("eligible_coverage", v25_result.get("coverage", 0.0)))
    ref_adj25 = float(v25_result.get("referee_adjustment_points", 0.0))

    a, b, c, d = st.columns(4)
    a.metric("V2.5 projected margin", margin_label)
    b.metric("Vs market spread", f"{edge_pts:+.2f} pts")
    c.metric("Eligible live coverage", f"{eligible_cov:.0%}")
    d.metric("Status", "2026 forward test")

    st.markdown("##### Moneyline")
    a, b, c, d = st.columns(4)
    a.metric(f"{home} win probability", f"{v25_result['p_home_win']:.1%}")
    b.metric(f"{home} ML EV", f"{v25_result['home_ml_ev']:+.1%}")
    c.metric(f"{away} win probability", f"{v25_result['p_away_win']:.1%}")
    d.metric(f"{away} ML EV", f"{v25_result['away_ml_ev']:+.1%}")

    st.markdown("##### Point spread")
    a, b, c, d = st.columns(4)
    a.metric(f"{home} {home_spread:+.1f} cover probability", f"{v25_result['p_home_cover']:.1%}")
    b.metric(f"{home} spread EV", f"{v25_result['home_spread_ev']:+.1%}")
    c.metric(f"{away} {-home_spread:+.1f} cover probability", f"{v25_result['p_away_cover']:.1%}")
    d.metric(f"{away} spread EV", f"{v25_result['away_spread_ev']:+.1%}")

    fam = v25_result.get("families", {}) or {}
    loaded = [k.upper() for k, v in fam.items() if v]
    missing_fam = [k.upper() for k, v in fam.items() if not v]
    st.caption("Live families loaded: " + (", ".join(loaded) if loaded else "none") + (" · unavailable/imputed: " + ", ".join(missing_fam) if missing_fam else ""))
    if abs(ref_adj25) >= 0.01:
        st.caption(f"V2.5 includes the same tiny leakage-safe referee overlay used by V2.4: {ref_adj25:+.2f} points.")

    unexpected25 = len(v25_result.get("unexpected_missing", []) or [])
    structural25 = len(v25_result.get("structural_missing", []) or [])
    if unexpected25:
        st.warning(
            f"V2.5 has {unexpected25} unexpectedly missing live features beyond {structural25} structurally unavailable features. "
            "Large probability/EV gaps should be treated cautiously until those feeds are fresh."
        )

    with st.expander("Why V2.5 is different / development results"):
        st.markdown(
            """
- **Margin engine:** 75% robust L1 LightGBM + 25% L2 LightGBM. The second component is deliberately shrunk toward the market to reduce overreaction.
- **Live inputs:** market/context + play-by-play team/matchup/possession data + QB weekly stats + ESPN QBR + Next Gen Stats + snaps + depth + live injuries.
- **Probability calibration:** moneyline and cover probabilities are calibrated separately from historical walk-forward predictions instead of assuming the raw margin distribution is perfectly normal.
- **2021–2025 development walk-forward:** **9.245 margin MAE** vs **9.762 for the closing spread**, an average improvement of about **0.52 points/game** across 1,424 games.
- The model beat the market MAE in each individual development season from 2021 through 2025. Bootstrap resampling of the historical MAE improvement was roughly **+0.38 to +0.66 points/game (95% interval)**.
- Historical ATS direction was about **59.9%** across non-pushes, but that is development evidence, **not an expectation for future betting performance**.
- **2026 is the real forward test.** V2.2/V2.3/V2.4 stay visible so V2.5 cannot silently replace weaker results after the fact.
            """
        )

st.markdown("---")

# -----------------------------
# V2.4 Challenger — head referee tracker + V2.3
# -----------------------------
st.subheader("V2.4 Challenger — Head Referee Layer")
st.caption(
    "V2.4 keeps V2.3 intact and adds a deliberately small head-referee overlay. The tracker uses only games completed "
    "before the selected matchup, shrinks small samples toward neutral, and caps the referee adjustment at ±0.25 points. "
    "Team/QB/injury inputs still update exactly as they do in V2.3."
)

v24_result = None
v24_error = None
auto_ref_info = {"referee": "", "source": "unavailable", "url": ""}
if resolve_head_referee is not None:
    try:
        auto_ref_info = resolve_head_referee(
            schedule=schedule,
            game=game,
            season=season,
            week=week,
            away=away,
            home=home,
        ) or auto_ref_info
    except Exception:
        pass

auto_ref = str(auto_ref_info.get("referee", "") or "")
ref_override = st.text_input(
    "Head referee",
    value=auto_ref,
    help="Auto-detected from nflverse/Football Zebras when available. You can correct it manually if a late assignment changes.",
    key=f"v24_ref_{season}_{week}_{away}_{home}",
)
ref_source = auto_ref_info.get("source", "unavailable")
ref_url = auto_ref_info.get("url", "")
if ref_url:
    st.caption(f"Assignment source: {ref_source} · {ref_url}")
else:
    st.caption(f"Assignment source: {ref_source}")

if build_side_prediction_v24 is None:
    v24_error = f"V2.4 module could not load: {V24_SIDE_IMPORT_ERROR}"
else:
    try:
        with st.spinner("Running V2.4 referee challenger…"):
            v24_result = build_side_prediction_v24(
                root=ROOT,
                schedule=schedule,
                game=game,
                prev_pbp=prev_pbp,
                cur_pbp=cur_pbp,
                home=home,
                away=away,
                season=season,
                week=week,
                market_total=float(market_total),
                over_odds=float(over_odds),
                under_odds=float(under_odds),
                home_spread=float(home_spread),
                home_spread_odds=float(home_spread_odds),
                away_spread_odds=float(away_spread_odds),
                home_ml=float(home_ml),
                away_ml=float(away_ml),
                dome=float(dome),
                referee_override=(ref_override.strip() if ref_override.strip() and ref_override.strip().casefold() != auto_ref.strip().casefold() else None),
            )
    except Exception as exc:
        v24_error = str(exc)

if v24_error:
    st.error(f"V2.4 challenger could not run: {v24_error}")

if v24_result:
    margin = float(v24_result["predicted_margin"])
    margin_label = f"{home} by {margin:.1f}" if margin >= 0 else f"{away} by {abs(margin):.1f}"
    rtrack = v24_result.get("referee_tracker", {}) or {}
    rassn = v24_result.get("referee_assignment", {}) or {}
    ref_name = rassn.get("referee", "") or rtrack.get("referee", "") or "Not found"
    ref_adj = float(v24_result.get("referee_adjustment_points", 0.0))

    a, b, c, d = st.columns(4)
    a.metric("V2.4 adjusted margin", margin_label)
    b.metric("Head referee", ref_name)
    c.metric("Referee adjustment", f"{ref_adj:+.2f} pts")
    d.metric("Status", "2026 forward test")

    st.markdown("##### Moneyline")
    a, b, c, d = st.columns(4)
    a.metric(f"{home} V2.4 win probability", f"{v24_result['p_home_win']:.1%}")
    b.metric(f"{home} ML EV", f"{v24_result['home_ml_ev']:+.1%}")
    c.metric(f"{away} V2.4 win probability", f"{v24_result['p_away_win']:.1%}")
    d.metric(f"{away} ML EV", f"{v24_result['away_ml_ev']:+.1%}")

    st.markdown("##### Point spread")
    a, b, c, d = st.columns(4)
    a.metric(f"{home} {home_spread:+.1f} cover probability", f"{v24_result['p_home_cover']:.1%}")
    b.metric(f"{home} spread EV", f"{v24_result['home_spread_ev']:+.1%}")
    c.metric(f"{away} {-home_spread:+.1f} cover probability", f"{v24_result['p_away_cover']:.1%}")
    d.metric(f"{away} spread EV", f"{v24_result['away_spread_ev']:+.1%}")

    with st.expander("Head referee tracker"):
        if rtrack.get("available"):
            def _ats_text(rec):
                rec = rec or {}
                return f"{rec.get('wins',0)}-{rec.get('losses',0)}-{rec.get('pushes',0)}"
            def _ou_text(rec):
                rec = rec or {}
                return f"{rec.get('over',0)}-{rec.get('under',0)}-{rec.get('pushes',0)}"

            a, b, c, d = st.columns(4)
            a.metric("Career games tracked", int(rtrack.get("career_games", 0)))
            b.metric("Last 3 seasons", int(rtrack.get("recent3_games", 0)))
            c.metric("Current season before game", int(rtrack.get("current_season_games", 0)))
            d.metric("Last 16 games", int(rtrack.get("last16_games", 0)))

            role_label = rtrack.get("role_label", "Same market role")
            a, b, c, d = st.columns(4)
            a.metric(f"{role_label} ATS", _ats_text(rtrack.get("role_ats")))
            b.metric("Career home-team ATS", _ats_text(rtrack.get("career_home_ats")))
            c.metric("Last-16 home-team ATS", _ats_text(rtrack.get("last16_home_ats")))
            d.metric("Career O/U/P", _ou_text(rtrack.get("career_ou")))

            a, b, c = st.columns(3)
            raw_role = rtrack.get("role_mean_ats_residual")
            raw_last = rtrack.get("last16_mean_ats_residual")
            signal = float(rtrack.get("signal_points", 0.0))
            a.metric("Same-role avg ATS residual", "—" if raw_role is None or pd.isna(raw_role) else f"{float(raw_role):+.2f} pts")
            b.metric("Last-16 avg ATS residual", "—" if raw_last is None or pd.isna(raw_last) else f"{float(raw_last):+.2f} pts")
            c.metric("Shrunk referee signal", f"{signal:+.2f} pts")

            st.caption(
                "Positive ATS residual means home teams historically beat the closing spread by more points; negative means the away side did. "
                "V2.4 combines career, same market-role, current-season and last-16 residuals with strong shrinkage, then caps the actual model move at ±0.25 points."
            )
        else:
            st.warning(rtrack.get("reason", "Referee history unavailable, so V2.4 applies no referee adjustment."))

        st.info(
            "Development check only (not a new untouched validation): applying this small referee layer to the 2021–2025 V2.3 walk-forward predictions "
            "changed margin MAE from about 9.580 to 9.578 and ATS side accuracy from about 55.36% to 55.79%. The difference is small, so 2026 forward results decide whether the layer stays."
        )

    if abs(ref_adj) < 0.01:
        st.caption("The referee layer is neutral for this matchup, so V2.4 will be nearly identical to V2.3.")
    else:
        direction = home if ref_adj > 0 else away
        st.caption(f"Referee layer leans slightly toward {direction}; the adjustment is intentionally limited to {abs(ref_adj):.2f} points.")

# -----------------------------
# V2.3 Challenger — live injuries + calibrated probabilities
# -----------------------------
st.subheader("V2.3 Challenger — Spread & Moneyline")
st.caption(
    "V2.3 keeps the V2.2 estimators frozen, adds a free multi-source injury layer (official NFL.com + ESPN + Sleeper) "
    "that matches the historical injury features, and applies conservative probability/margin calibration learned from the 2021–2025 "
    "walk-forward predictions. V2.2 remains below as the untouched benchmark."
)

v23_result = None
v23_error = None
if build_side_prediction_v23 is None:
    v23_error = f"V2.3 module could not load: {V23_SIDE_IMPORT_ERROR}"
else:
    try:
        with st.spinner("Loading free injury sources and running V2.3 challenger…"):
            v23_result = build_side_prediction_v23(
                root=ROOT,
                schedule=schedule,
                game=game,
                prev_pbp=prev_pbp,
                cur_pbp=cur_pbp,
                home=home,
                away=away,
                season=season,
                week=week,
                market_total=float(market_total),
                over_odds=float(over_odds),
                under_odds=float(under_odds),
                home_spread=float(home_spread),
                home_spread_odds=float(home_spread_odds),
                away_spread_odds=float(away_spread_odds),
                home_ml=float(home_ml),
                away_ml=float(away_ml),
                dome=float(dome),
            )
    except Exception as exc:
        v23_error = str(exc)

if v23_error:
    st.error(f"V2.3 challenger could not run: {v23_error}")

if v23_result:
    margin = float(v23_result["predicted_margin"])
    margin_label = f"{home} by {margin:.1f}" if margin >= 0 else f"{away} by {abs(margin):.1f}"

    a, b, c, d = st.columns(4)
    a.metric("V2.3 calibrated margin", margin_label)
    b.metric("Sportsbook home spread", f"{home_spread:+.1f}")
    c.metric("Eligible live coverage", f"{v23_result['eligible_coverage']:.0%}")
    d.metric("Status", "2026 challenger test")

    st.markdown("##### Moneyline")
    a, b, c, d = st.columns(4)
    a.metric(f"{home} calibrated win probability", f"{v23_result['p_home_win']:.1%}")
    b.metric(f"{home} ML EV", f"{v23_result['home_ml_ev']:+.1%}")
    c.metric(f"{away} calibrated win probability", f"{v23_result['p_away_win']:.1%}")
    d.metric(f"{away} ML EV", f"{v23_result['away_ml_ev']:+.1%}")

    a, b, c, d = st.columns(4)
    a.metric(f"{home} fair ML", fmt_odds(v23_result["fair_home_ml"]))
    b.metric(f"{home} offered ML", fmt_odds(home_ml))
    c.metric(f"{away} fair ML", fmt_odds(v23_result["fair_away_ml"]))
    d.metric(f"{away} offered ML", fmt_odds(away_ml))

    st.markdown("##### Point spread")
    a, b, c, d = st.columns(4)
    a.metric(f"{home} {home_spread:+.1f} calibrated cover probability", f"{v23_result['p_home_cover']:.1%}")
    b.metric(f"{home} spread EV", f"{v23_result['home_spread_ev']:+.1%}")
    c.metric(f"{away} {-home_spread:+.1f} calibrated cover probability", f"{v23_result['p_away_cover']:.1%}")
    d.metric(f"{away} spread EV", f"{v23_result['away_spread_ev']:+.1%}")

    a, b, c, d = st.columns(4)
    a.metric(f"{home} fair spread odds", fmt_odds(v23_result["fair_home_spread"]))
    b.metric(f"{home} offered odds", fmt_odds(home_spread_odds))
    c.metric(f"{away} fair spread odds", fmt_odds(v23_result["fair_away_spread"]))
    d.metric(f"{away} offered odds", fmt_odds(away_spread_odds))

    fam = v23_result.get("families", {})
    injury_ok = bool(fam.get("injuries"))
    structural_n = len(v23_result.get("structural_missing", []))
    unexpected_n = len(v23_result.get("unexpected_missing", []))
    st.caption(
        f"Raw populated coverage: {v23_result['coverage']:.1%} of {v23_result['selected_feature_count']} selected features. "
        f"{structural_n} features are structurally unavailable before Week {week} because they require a prior current-season "
        f"observation; those same fields were missing and imputed in historical Week 1 training rows. "
        f"Unexpected missing selected features: {unexpected_n}. Injury feed: {'loaded' if injury_ok else 'unavailable'}. "
        f"Source: {v23_result.get('injury_meta', {}).get('source', 'none')}."
    )

    if v23_result["eligible_coverage"] < 0.90 or not injury_ok:
        st.warning(
            "V2.3 is missing live inputs that should normally be available. Treat large EV estimates cautiously until the "
            "missing feed is restored."
        )
    else:
        st.info(
            "V2.3 is a challenger, not a replacement for V2.2 yet. Its calibration improved historical out-of-sample "
            "spread MAE and moneyline probability scoring, but the 2026 forward test is what decides whether it is better live."
        )

    with st.expander("Free injury + breaking-news inputs"):
        imeta = v23_result.get("injury_meta", {}) or {}
        st.write(f"**Injury sources used:** {imeta.get('source', 'none')}")
        detail = imeta.get("source_detail", {}) or {}
        if detail:
            rows=[]
            for tm, vals in detail.items():
                rows.append({"Team": tm, "NFL.com": vals.get("official",0), "ESPN": vals.get("espn",0), "Sleeper": vals.get("sleeper",0), "Merged players": vals.get("merged",0)})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        moved = imeta.get("official_team_reassignments", []) or []
        if moved:
            st.warning(
                f"Safety check repaired {len(moved)} NFL.com player/team assignments after cross-checking the current Sleeper roster. "
                "This prevents a scraped table from being attached to the wrong team."
            )

        player_map = imeta.get("players", {}) or {}
        injury_rows=[]
        for tm in [home, away]:
            for r in player_map.get(tm, []) or []:
                if r.get("status") or r.get("practice"):
                    injury_rows.append({
                        "Team":tm, "Player":r.get("player",""), "Pos":r.get("position",""),
                        "Game status":r.get("status","") or "—", "Practice":r.get("practice","") or "—",
                        "Injury":r.get("injury","") or "—", "Source":r.get("source","")
                    })
        if injury_rows:
            st.dataframe(pd.DataFrame(injury_rows), width="stretch", hide_index=True)
        else:
            st.caption("No player-level injury rows were returned for this matchup.")

        st.markdown("**Breaking-news watch (free Google News RSS)**")
        st.caption("Headlines are shown for review only and do not automatically override the model. This avoids a vague or misquoted headline changing a bet.")
        if get_free_breaking_news is not None:
            try:
                news_items = get_free_breaking_news(home, away, 8)
            except Exception:
                news_items = []
        else:
            news_items = []
        if news_items:
            for nitem in news_items:
                title=nitem.get("title","Breaking NFL update")
                link=nitem.get("link","")
                src=nitem.get("source","")
                pub=nitem.get("published","")
                if link:
                    st.markdown(f"- [{title}]({link}) — {src} {('· ' + pub) if pub else ''}")
                else:
                    st.markdown(f"- {title} — {src} {('· ' + pub) if pub else ''}")
        else:
            st.caption("No recent injury-related headlines found for this matchup right now.")

    with st.expander("What changed from V2.2?"):
        st.markdown(
            """
            - **Current injuries:** free consensus from **NFL.com official reports + ESPN + Sleeper** fills the same OUT / DOUBTFUL / QUESTIONABLE and position-group fields used in historical training. NFL.com is preferred when available.
            - **Spread margin calibration:** V2.3 keeps **55%** of V2.2's correction away from the market line. This reduced 2021–2025 walk-forward spread MAE from about **9.67 to 9.58** points; the closing market was about **9.76**.
            - **Spread probability calibration:** cover probabilities are compressed toward 50% to reduce overconfidence.
            - **Moneyline calibration:** **90% V2.2 probability + 10% de-vigged market probability**. Historical Brier score improved slightly from about **0.20384 to 0.20372**; the market was about **0.21155**.
            - **V2.2 is still shown below unchanged** so we can compare both versions prospectively in 2026.
            """
        )


# -----------------------------
# V2.2 Spread & Moneyline Model
# -----------------------------
st.subheader("V2.2 Frozen Benchmark")
st.caption(
    "This is the frozen V2.2 side model being forward-tested on 2026 games. "
    "It predicts the game margin, then converts that margin distribution into moneyline win probability and spread cover probability."
)

v2_result = None
v2_error = None
if build_side_prediction is None:
    v2_error = f"V2.2 module could not load: {V22_SIDE_IMPORT_ERROR}"
else:
    try:
        with st.spinner("Building V2.2 live side features and running the frozen model…"):
            v2_result = build_side_prediction(
                root=ROOT,
                schedule=schedule,
                game=game,
                prev_pbp=prev_pbp,
                cur_pbp=cur_pbp,
                home=home,
                away=away,
                season=season,
                week=week,
                market_total=float(market_total),
                over_odds=float(over_odds),
                under_odds=float(under_odds),
                home_spread=float(home_spread),
                home_spread_odds=float(home_spread_odds),
                away_spread_odds=float(away_spread_odds),
                home_ml=float(home_ml),
                away_ml=float(away_ml),
                dome=float(dome),
            )
    except Exception as exc:
        v2_error = str(exc)

if v2_error:
    st.error(f"V2.2 side model could not run: {v2_error}")
    st.caption("The older market-consensus comparison is still available below as a fallback.")

if v2_result:
    margin = float(v2_result["predicted_margin"])
    margin_label = f"{home} by {margin:.1f}" if margin >= 0 else f"{away} by {abs(margin):.1f}"
    a, b, c, d = st.columns(4)
    a.metric("V2.2 projected margin", margin_label)
    b.metric("Sportsbook home spread", f"{home_spread:+.1f}")
    c.metric("Raw feature coverage", f"{v2_result['coverage']:.0%}")
    d.metric("Status", "2026 forward test")

    st.markdown("##### Moneyline")
    a, b, c, d = st.columns(4)
    a.metric(f"{home} win probability", f"{v2_result['p_home_win']:.1%}")
    b.metric(f"{home} ML EV", f"{v2_result['home_ml_ev']:+.1%}")
    c.metric(f"{away} win probability", f"{v2_result['p_away_win']:.1%}")
    d.metric(f"{away} ML EV", f"{v2_result['away_ml_ev']:+.1%}")

    a, b, c, d = st.columns(4)
    a.metric(f"{home} fair ML", fmt_odds(v2_result["fair_home_ml"]))
    b.metric(f"{home} offered ML", fmt_odds(home_ml))
    c.metric(f"{away} fair ML", fmt_odds(v2_result["fair_away_ml"]))
    d.metric(f"{away} offered ML", fmt_odds(away_ml))

    st.markdown("##### Point spread")
    a, b, c, d = st.columns(4)
    a.metric(f"{home} {home_spread:+.1f} cover probability", f"{v2_result['p_home_cover']:.1%}")
    b.metric(f"{home} spread EV", f"{v2_result['home_spread_ev']:+.1%}")
    c.metric(f"{away} {-home_spread:+.1f} cover probability", f"{v2_result['p_away_cover']:.1%}")
    d.metric(f"{away} spread EV", f"{v2_result['away_spread_ev']:+.1%}")

    a, b, c, d = st.columns(4)
    a.metric(f"{home} fair spread odds", fmt_odds(v2_result["fair_home_spread"]))
    b.metric(f"{home} offered odds", fmt_odds(home_spread_odds))
    c.metric(f"{away} fair spread odds", fmt_odds(v2_result["fair_away_spread"]))
    d.metric(f"{away} offered odds", fmt_odds(away_spread_odds))

    fam = v2_result.get("families", {})
    loaded_names = [k.upper() for k, ok in fam.items() if ok]
    missing_names = [k.upper() for k, ok in fam.items() if not ok]
    st.caption(
        f"Selected-feature coverage: {v2_result['coverage']:.1%} of "
        f"{v2_result['selected_feature_count']} features selected by the frozen models are populated live. "
        f"Loaded families: {', '.join(loaded_names) if loaded_names else 'none'}. "
        f"Imputed/unavailable families: {', '.join(missing_names) if missing_names else 'none'}."
    )
    structural_n = len(v2_result.get("structural_missing", []))
    unexpected_n = len(v2_result.get("unexpected_missing", []))
    if unexpected_n > 0:
        st.warning(
            f"V2.2 has {unexpected_n} selected live features missing beyond the {structural_n} structurally unavailable "
            "early-season fields. This benchmark intentionally does not use the new ESPN injury feed."
        )
    else:
        st.info(
            "These are frozen-model forward-test probabilities, not a guarantee of profit. "
            "The 2026 results should be tracked without retuning V2.2."
        )

# Keep the market-consensus comparison available as a useful independent check.
selected_key = selected_book_row.get("key") if selected_book_row else None
ml_home_p, ml_away_p, ml_books, ml_used_selected = consensus_moneyline(book_rows, selected_key=selected_key)
sp_home_p, sp_away_p, sp_books, sp_used_selected = consensus_spread(book_rows, float(home_spread), selected_key=selected_key)

with st.expander("Compare V2.2 with sportsbook consensus"):
    if book_rows and ml_home_p is not None:
        st.markdown("**Moneyline consensus**")
        a, b = st.columns(2)
        a.metric(f"{home} consensus fair win %", f"{ml_home_p:.1%}")
        b.metric(f"{away} consensus fair win %", f"{ml_away_p:.1%}")
        if v2_result:
            st.caption(
                f"V2.2 vs consensus: {home} {v2_result['p_home_win'] - ml_home_p:+.1%} probability difference."
            )
    else:
        st.caption("No complete moneyline consensus is available from the current Odds API response.")

    if book_rows and sp_home_p is not None:
        st.markdown("**Exact-line spread consensus**")
        a, b = st.columns(2)
        a.metric(f"{home} {home_spread:+.1f} consensus cover %", f"{sp_home_p:.1%}")
        b.metric(f"{away} {-home_spread:+.1f} consensus cover %", f"{sp_away_p:.1%}")
        if v2_result:
            st.caption(
                f"V2.2 vs consensus: {home} spread probability difference "
                f"{v2_result['p_home_cover'] - sp_home_p:+.1%}."
            )
    else:
        st.caption("No other-book consensus is available at this exact spread number.")

st.divider()
with st.expander("Data sources & refresh behavior"):
    st.markdown(
        """
- **Schedule / game metadata:** nflverse `games.csv`
- **V1 totals inputs:** calculated from nflverse play-by-play
- **V2.2/V2.5 spread & ML inputs:** nflverse play-by-play, weekly player/QB stats, ESPN QBR, Next Gen Stats, snap counts, depth charts, live injuries, and live market prices when available
- **Weather:** Open-Meteo stadium-area forecast (used by V1 totals; V2.2 side model was trained without observed game-weather leakage)
- **Sportsbook totals / spreads / moneylines / player props / multi-book prices:** The Odds API when a key is configured
- **Fallback line:** nflverse schedule snapshot or manual override
- Schedule refresh cache: ~30 minutes
- Weather refresh cache: ~30 minutes
- Odds refresh cache: ~2 minutes
- Play-by-play refresh cache: ~1 hour
        """
    )

with st.expander("Model status"):
    st.caption(
        f"V1 historical holdout: {int(HOLDOUT['Wins'])}-{int(HOLDOUT['Losses'])}, "
        f"{float(HOLDOUT['Win_Pct']):.1%} wins, {float(HOLDOUT['ROI']):.1%} ROI."
    )
    st.caption("V2.5 spread/moneyline challenger: LightGBM residual ensemble trained on 2018–2025 only; 2026 is the clean forward test. V2.2/V2.3/V2.4 remain visible as frozen benchmarks. Totals remain on V1 while the separate totals R&D continues.")

