from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd

PROP_MARKET_LABELS = {
    "player_pass_yds": "Passing yards",
    "player_pass_tds": "Passing TDs",
    "player_pass_completions": "Pass completions",
    "player_pass_attempts": "Pass attempts",
    "player_pass_interceptions": "Pass interceptions",
    "player_pass_rush_yds": "Pass + rush yards",
    "player_rush_yds": "Rushing yards",
    "player_rush_attempts": "Rush attempts",
    "player_receptions": "Receptions",
    "player_reception_yds": "Receiving yards",
    "player_rush_reception_yds": "Rush + receiving yards",
    "player_tds": "Touchdowns",
    "player_anytime_td": "Anytime TD",
}

CORE_PROP_MARKETS = [
    "player_pass_yds",
    "player_pass_tds",
    "player_pass_completions",
    "player_pass_interceptions",
    "player_rush_yds",
    "player_rush_attempts",
    "player_receptions",
    "player_reception_yds",
]


def american_to_decimal(odds):
    if odds is None:
        return None
    try:
        o = float(odds)
    except Exception:
        return None
    if not math.isfinite(o) or o == 0:
        return None
    return 1.0 + (o / 100.0 if o > 0 else 100.0 / abs(o))


def implied_prob(odds):
    d = american_to_decimal(odds)
    return None if d is None or d <= 1 else 1.0 / d


def fair_american(p):
    if p is None or not (0 < float(p) < 1):
        return None
    p = float(p)
    return -100.0 * p / (1.0 - p) if p >= 0.5 else 100.0 * (1.0 - p) / p


def ev_from_prob(prob, odds):
    d = american_to_decimal(odds)
    if prob is None or d is None:
        return None
    return float(prob) * d - 1.0


def devig_two_way(odds_a, odds_b):
    pa, pb = implied_prob(odds_a), implied_prob(odds_b)
    if pa is None or pb is None or pa + pb <= 0:
        return None, None
    z = pa + pb
    return pa / z, pb / z


def best_offer(rows):
    """Return row with highest decimal price."""
    best = None
    best_d = -1.0
    for r in rows:
        d = american_to_decimal(r.get("Odds"))
        if d is not None and d > best_d:
            best_d, best = d, r
    return best


def complement_side(side: str):
    s = str(side).strip().lower()
    if s == "over":
        return "Under"
    if s == "under":
        return "Over"
    if s == "yes":
        return "No"
    if s == "no":
        return "Yes"
    return None


def parse_prop_event(payload: dict) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    matchup = f"{payload.get('away_team', '')} @ {payload.get('home_team', '')}".strip()
    rows = []
    for b in payload.get("bookmakers", []) or []:
        bkey = b.get("key")
        bname = b.get("title") or bkey
        for m in b.get("markets", []) or []:
            mkey = m.get("key")
            mlabel = PROP_MARKET_LABELS.get(mkey, mkey)
            updated = m.get("last_update") or b.get("last_update")
            for o in m.get("outcomes", []) or []:
                player = o.get("description")
                side = o.get("name")
                odds = o.get("price")
                point = o.get("point")
                if not player or side is None or odds is None:
                    continue
                try:
                    odds = float(odds)
                except Exception:
                    continue
                try:
                    point = float(point) if point is not None else None
                except Exception:
                    point = None
                rows.append({
                    "Event ID": payload.get("id"),
                    "Matchup": matchup,
                    "Commence": payload.get("commence_time"),
                    "Market Key": mkey,
                    "Market": mlabel,
                    "Player": str(player),
                    "Side": str(side),
                    "Line": point,
                    "Odds": odds,
                    "Book Key": bkey,
                    "Book": bname,
                    "Updated": updated,
                })
    return pd.DataFrame(rows)


def combine_prop_payloads(payloads) -> pd.DataFrame:
    frames = [parse_prop_event(x) for x in (payloads or [])]
    frames = [x for x in frames if not x.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def prop_value_table(payloads, minimum_other_books: int = 1) -> pd.DataFrame:
    """
    Consensus EV at the EXACT same player/market/line.
    Best offered price is evaluated against de-vigged two-way probabilities from OTHER books.
    At least one other full two-way book is required, which avoids self-confirming EV.
    """
    offers = combine_prop_payloads(payloads)
    if offers.empty:
        return pd.DataFrame()

    key_cols = ["Event ID", "Matchup", "Market Key", "Market", "Player", "Line"]
    out = []
    for _, g in offers.groupby(key_cols, dropna=False):
        # Only markets with a recognized complementary side can be fairly de-vigged.
        for side in sorted(g["Side"].dropna().unique()):
            comp = complement_side(side)
            if comp is None:
                continue
            side_rows = g[g["Side"].str.lower().eq(str(side).lower())].to_dict("records")
            if not side_rows:
                continue
            candidate = best_offer(side_rows)
            if candidate is None:
                continue

            fair_probs = []
            comparison_books = []
            for bkey, bg in g.groupby("Book Key", dropna=False):
                if bkey == candidate.get("Book Key"):
                    continue
                a = bg[bg["Side"].str.lower().eq(str(side).lower())]
                b = bg[bg["Side"].str.lower().eq(comp.lower())]
                if a.empty or b.empty:
                    continue
                # Same exact line is already guaranteed by the outer group.
                oa = best_offer(a.to_dict("records"))
                ob = best_offer(b.to_dict("records"))
                if oa is None or ob is None:
                    continue
                pa, _ = devig_two_way(oa["Odds"], ob["Odds"])
                if pa is not None:
                    fair_probs.append(pa)
                    comparison_books.append(str(oa.get("Book")))

            if len(fair_probs) < max(1, int(minimum_other_books)):
                continue

            p = sum(fair_probs) / len(fair_probs)
            ev = ev_from_prob(p, candidate["Odds"])
            implied = implied_prob(candidate["Odds"])
            row = dict(candidate)
            row.update({
                "Fair Prob": p,
                "Fair Odds": fair_american(p),
                "Implied Prob": implied,
                "EV": ev,
                "Probability Edge": p - implied if implied is not None else None,
                "Consensus Books": len(fair_probs),
                "Consensus Sources": ", ".join(sorted(set(comparison_books))),
            })
            out.append(row)

    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    # Keep one row per exact selection (the best offered price).
    sel = ["Event ID", "Market Key", "Player", "Line", "Side"]
    df["_decimal"] = df["Odds"].map(american_to_decimal)
    df = df.sort_values(["EV", "Consensus Books", "_decimal"], ascending=[False, False, False])
    df = df.drop_duplicates(sel, keep="first").drop(columns=["_decimal"])
    return df.reset_index(drop=True)


def _distinct_best_pair(a_rows, b_rows):
    """Best two-price combination, preferring different books."""
    combos = []
    for a in a_rows:
        da = american_to_decimal(a.get("Odds"))
        if da is None:
            continue
        for b in b_rows:
            db = american_to_decimal(b.get("Odds"))
            if db is None:
                continue
            if a.get("Book Key") == b.get("Book Key"):
                continue
            s = 1.0 / da + 1.0 / db
            combos.append((s, a, b))
    if not combos:
        # Same-book negative-hold is extremely unusual, but keep it visible if it exists.
        for a in a_rows:
            da = american_to_decimal(a.get("Odds"))
            if da is None:
                continue
            for b in b_rows:
                db = american_to_decimal(b.get("Odds"))
                if db is None:
                    continue
                s = 1.0 / da + 1.0 / db
                combos.append((s, a, b))
    return min(combos, key=lambda x: x[0]) if combos else None


def arb_math(odds_a, odds_b, total_stake=100.0):
    da, db = american_to_decimal(odds_a), american_to_decimal(odds_b)
    if da is None or db is None:
        return None
    inv = 1.0 / da + 1.0 / db
    if inv <= 0:
        return None
    stake_a = float(total_stake) * (1.0 / da) / inv
    stake_b = float(total_stake) * (1.0 / db) / inv
    payout = float(total_stake) / inv
    profit = payout - float(total_stake)
    return {
        "Implied Sum": inv,
        "ROI": 1.0 / inv - 1.0,
        "Stake A": stake_a,
        "Stake B": stake_b,
        "Guaranteed Payout": payout,
        "Guaranteed Profit": profit,
    }


def prop_arbitrage_table(payloads, total_stake=100.0, min_roi=0.0) -> pd.DataFrame:
    offers = combine_prop_payloads(payloads)
    if offers.empty:
        return pd.DataFrame()
    key_cols = ["Event ID", "Matchup", "Market Key", "Market", "Player", "Line"]
    rows = []
    for _, g in offers.groupby(key_cols, dropna=False):
        done = set()
        for side in g["Side"].dropna().unique():
            comp = complement_side(side)
            if comp is None:
                continue
            pair_key = tuple(sorted([side.lower(), comp.lower()]))
            if pair_key in done:
                continue
            done.add(pair_key)
            a = g[g["Side"].str.lower().eq(str(side).lower())].to_dict("records")
            b = g[g["Side"].str.lower().eq(comp.lower())].to_dict("records")
            pair = _distinct_best_pair(a, b)
            if pair is None:
                continue
            inv, ra, rb = pair
            if inv >= 1.0:
                continue
            calc = arb_math(ra["Odds"], rb["Odds"], total_stake)
            if calc is None or calc["ROI"] < float(min_roi):
                continue
            rows.append({
                "Matchup": ra["Matchup"], "Market": ra["Market"], "Player": ra["Player"], "Line": ra["Line"],
                "Side A": ra["Side"], "Odds A": ra["Odds"], "Book A": ra["Book"],
                "Side B": rb["Side"], "Odds B": rb["Odds"], "Book B": rb["Book"],
                **calc,
            })
    return pd.DataFrame(rows).sort_values("ROI", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def _core_book_rows(event):
    rows = []
    if not event:
        return rows
    home, away = event.get("home_team"), event.get("away_team")
    matchup = f"{away} @ {home}"
    for b in event.get("bookmakers", []) or []:
        bk, bn = b.get("key"), b.get("title") or b.get("key")
        for m in b.get("markets", []) or []:
            key = m.get("key")
            if key not in {"h2h", "spreads", "totals"}:
                continue
            for o in m.get("outcomes", []) or []:
                try:
                    price = float(o.get("price"))
                except Exception:
                    continue
                try:
                    point = float(o.get("point")) if o.get("point") is not None else None
                except Exception:
                    point = None
                rows.append({
                    "Event ID": event.get("id"), "Matchup": matchup, "Home": home, "Away": away,
                    "Market Key": key, "Side": o.get("name"), "Line": point, "Odds": price,
                    "Book Key": bk, "Book": bn, "Updated": m.get("last_update") or b.get("last_update"),
                })
    return rows


def core_best_lines(odds_events) -> pd.DataFrame:
    rows = []
    for event in odds_events or []:
        offers = pd.DataFrame(_core_book_rows(event))
        if offers.empty:
            continue
        home, away = event.get("home_team"), event.get("away_team")
        matchup = f"{away} @ {home}"

        # ML: price is all that matters.
        for team in [away, home]:
            x = offers[(offers["Market Key"] == "h2h") & (offers["Side"] == team)]
            if not x.empty:
                rr = best_offer(x.to_dict("records"))
                rows.append({"Matchup": matchup, "Market": "Moneyline", "Selection": team, "Line": None, "Odds": rr["Odds"], "Book": rr["Book"]})

        # Spread: most favorable point first, then best price at that point.
        for team in [away, home]:
            x = offers[(offers["Market Key"] == "spreads") & (offers["Side"] == team) & offers["Line"].notna()]
            if not x.empty:
                best_line = float(x["Line"].max())
                rr = best_offer(x[x["Line"] == best_line].to_dict("records"))
                rows.append({"Matchup": matchup, "Market": "Spread", "Selection": team, "Line": best_line, "Odds": rr["Odds"], "Book": rr["Book"]})

        # Totals: Over wants lowest number; Under wants highest number.
        for side, fn in [("Over", "min"), ("Under", "max")]:
            x = offers[(offers["Market Key"] == "totals") & (offers["Side"].str.lower() == side.lower()) & offers["Line"].notna()]
            if not x.empty:
                line = float(x["Line"].min() if fn == "min" else x["Line"].max())
                rr = best_offer(x[x["Line"] == line].to_dict("records"))
                rows.append({"Matchup": matchup, "Market": "Total", "Selection": side, "Line": line, "Odds": rr["Odds"], "Book": rr["Book"]})
    return pd.DataFrame(rows)


def core_arbitrage_table(odds_events, total_stake=100.0, min_roi=0.0) -> pd.DataFrame:
    arbs = []
    for event in odds_events or []:
        rows = _core_book_rows(event)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        home, away = event.get("home_team"), event.get("away_team")
        matchup = f"{away} @ {home}"

        # Moneyline
        a = df[(df["Market Key"] == "h2h") & (df["Side"] == home)].to_dict("records")
        b = df[(df["Market Key"] == "h2h") & (df["Side"] == away)].to_dict("records")
        pair = _distinct_best_pair(a, b)
        if pair and pair[0] < 1.0:
            calc = arb_math(pair[1]["Odds"], pair[2]["Odds"], total_stake)
            if calc and calc["ROI"] >= float(min_roi):
                arbs.append({"Matchup": matchup, "Market": "Moneyline", "Line": None,
                             "Side A": home, "Odds A": pair[1]["Odds"], "Book A": pair[1]["Book"],
                             "Side B": away, "Odds B": pair[2]["Odds"], "Book B": pair[2]["Book"], **calc})

        # Exact-line spreads
        home_lines = sorted(set(df[(df["Market Key"] == "spreads") & (df["Side"] == home)]["Line"].dropna()))
        for hl in home_lines:
            a = df[(df["Market Key"] == "spreads") & (df["Side"] == home) & (df["Line"] == hl)].to_dict("records")
            b = df[(df["Market Key"] == "spreads") & (df["Side"] == away) & (df["Line"] == -hl)].to_dict("records")
            pair = _distinct_best_pair(a, b)
            if pair and pair[0] < 1.0:
                calc = arb_math(pair[1]["Odds"], pair[2]["Odds"], total_stake)
                if calc and calc["ROI"] >= float(min_roi):
                    arbs.append({"Matchup": matchup, "Market": "Spread", "Line": float(hl),
                                 "Side A": f"{home} {hl:+.1f}", "Odds A": pair[1]["Odds"], "Book A": pair[1]["Book"],
                                 "Side B": f"{away} {-hl:+.1f}", "Odds B": pair[2]["Odds"], "Book B": pair[2]["Book"], **calc})

        # Exact-line totals
        totals = sorted(set(df[df["Market Key"] == "totals"]["Line"].dropna()))
        for line in totals:
            a = df[(df["Market Key"] == "totals") & (df["Side"].str.lower() == "over") & (df["Line"] == line)].to_dict("records")
            b = df[(df["Market Key"] == "totals") & (df["Side"].str.lower() == "under") & (df["Line"] == line)].to_dict("records")
            pair = _distinct_best_pair(a, b)
            if pair and pair[0] < 1.0:
                calc = arb_math(pair[1]["Odds"], pair[2]["Odds"], total_stake)
                if calc and calc["ROI"] >= float(min_roi):
                    arbs.append({"Matchup": matchup, "Market": "Total", "Line": float(line),
                                 "Side A": f"Over {line:.1f}", "Odds A": pair[1]["Odds"], "Book A": pair[1]["Book"],
                                 "Side B": f"Under {line:.1f}", "Odds B": pair[2]["Odds"], "Book B": pair[2]["Book"], **calc})

    return pd.DataFrame(arbs).sort_values("ROI", ascending=False).reset_index(drop=True) if arbs else pd.DataFrame()


def core_middle_table(odds_events) -> pd.DataFrame:
    rows = []
    for event in odds_events or []:
        df = pd.DataFrame(_core_book_rows(event))
        if df.empty:
            continue
        home, away = event.get("home_team"), event.get("away_team")
        matchup = f"{away} @ {home}"

        # Totals middle: Over lowest available total + Under highest available total.
        ov = df[(df["Market Key"] == "totals") & (df["Side"].str.lower() == "over") & df["Line"].notna()]
        un = df[(df["Market Key"] == "totals") & (df["Side"].str.lower() == "under") & df["Line"].notna()]
        if not ov.empty and not un.empty:
            lo, hi = float(ov["Line"].min()), float(un["Line"].max())
            if lo < hi:
                a = best_offer(ov[ov["Line"] == lo].to_dict("records"))
                b = best_offer(un[un["Line"] == hi].to_dict("records"))
                if a and b and a["Book Key"] != b["Book Key"]:
                    rows.append({"Matchup": matchup, "Type": "Total middle", "Window": f"{lo:.1f} to {hi:.1f}",
                                 "Bet A": f"Over {lo:.1f} ({int(a['Odds']):+d})", "Book A": a["Book"],
                                 "Bet B": f"Under {hi:.1f} ({int(b['Odds']):+d})", "Book B": b["Book"]})

        # Spread middle: highest home point + highest away point; overlap exists if their sum > 0.
        hs = df[(df["Market Key"] == "spreads") & (df["Side"] == home) & df["Line"].notna()]
        aws = df[(df["Market Key"] == "spreads") & (df["Side"] == away) & df["Line"].notna()]
        if not hs.empty and not aws.empty:
            hline, aline = float(hs["Line"].max()), float(aws["Line"].max())
            if hline + aline > 0:
                a = best_offer(hs[hs["Line"] == hline].to_dict("records"))
                b = best_offer(aws[aws["Line"] == aline].to_dict("records"))
                if a and b and a["Book Key"] != b["Book Key"]:
                    rows.append({"Matchup": matchup, "Type": "Spread middle", "Window": f"{hline + aline:.1f} pts overlap",
                                 "Bet A": f"{home} {hline:+.1f} ({int(a['Odds']):+d})", "Book A": a["Book"],
                                 "Bet B": f"{away} {aline:+.1f} ({int(b['Odds']):+d})", "Book B": b["Book"]})
    return pd.DataFrame(rows)

# ============================================================================
# Multi-sport Market Radar / anomaly detector
# ============================================================================

SPORT_KEYS = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb",
    "NHL": "icehockey_nhl",
}

FUTURE_SPORT_KEYS = {
    "NFL": "americanfootball_nfl_super_bowl_winner",
    "NBA": "basketball_nba_championship_winner",
    "MLB": "baseball_mlb_world_series_winner",
    "NHL": "icehockey_nhl_championship_winner",
}

EXCHANGE_BOOK_KEYS = {"novig", "kalshi", "polymarket", "prophetx"}

# Curated current The Odds API market keys. The scanner is intentionally split
# into core/deep/alternate presets so a free 500-credit plan can be conserved.
SPORT_PROP_MARKETS = {
    "NFL": {
        "player_pass_yds": "Passing yards",
        "player_pass_tds": "Passing TDs",
        "player_pass_completions": "Pass completions",
        "player_pass_attempts": "Pass attempts",
        "player_pass_interceptions": "Pass interceptions",
        "player_pass_longest_completion": "Longest completion",
        "player_pass_rush_yds": "Pass + rush yards",
        "player_rush_yds": "Rushing yards",
        "player_rush_attempts": "Rush attempts",
        "player_rush_longest": "Longest rush",
        "player_receptions": "Receptions",
        "player_reception_yds": "Receiving yards",
        "player_reception_longest": "Longest reception",
        "player_reception_tds": "Receiving TDs",
        "player_rush_reception_yds": "Rush + receiving yards",
        "player_rush_tds": "Rushing TDs",
        "player_anytime_td": "Anytime TD",
        "player_1st_td": "First TD scorer",
        "player_last_td": "Last TD scorer",
    },
    "NBA": {
        "player_points": "Points",
        "player_rebounds": "Rebounds",
        "player_assists": "Assists",
        "player_threes": "Three-pointers",
        "player_blocks": "Blocks",
        "player_steals": "Steals",
        "player_blocks_steals": "Blocks + steals",
        "player_turnovers": "Turnovers",
        "player_points_rebounds_assists": "Points + rebounds + assists",
        "player_points_rebounds": "Points + rebounds",
        "player_points_assists": "Points + assists",
        "player_rebounds_assists": "Rebounds + assists",
        "player_field_goals": "Field goals",
        "player_frees_made": "Free throws made",
        "player_frees_attempts": "Free throws attempted",
        "player_first_basket": "First basket scorer",
        "player_double_double": "Double-double",
        "player_triple_double": "Triple-double",
    },
    "MLB": {
        "batter_home_runs": "Batter home runs",
        "batter_hits": "Batter hits",
        "batter_total_bases": "Batter total bases",
        "batter_rbis": "Batter RBIs",
        "batter_runs_scored": "Batter runs",
        "batter_hits_runs_rbis": "Hits + runs + RBIs",
        "batter_singles": "Batter singles",
        "batter_doubles": "Batter doubles",
        "batter_triples": "Batter triples",
        "batter_walks": "Batter walks",
        "batter_strikeouts": "Batter strikeouts",
        "batter_stolen_bases": "Batter stolen bases",
        "pitcher_strikeouts": "Pitcher strikeouts",
        "pitcher_record_a_win": "Pitcher to record a win",
        "pitcher_hits_allowed": "Pitcher hits allowed",
        "pitcher_walks": "Pitcher walks",
        "pitcher_earned_runs": "Pitcher earned runs",
        "pitcher_outs": "Pitcher outs",
    },
    "NHL": {
        "player_points": "Points",
        "player_power_play_points": "Power-play points",
        "player_assists": "Assists",
        "player_blocked_shots": "Blocked shots",
        "player_shots_on_goal": "Shots on goal",
        "player_goals": "Goals",
        "player_total_saves": "Goalie saves",
        "player_goal_scorer_first": "First goal scorer",
        "player_goal_scorer_last": "Last goal scorer",
        "player_goal_scorer_anytime": "Anytime goal scorer",
    },
}

SPORT_CORE_PROP_KEYS = {
    "NFL": [
        "player_pass_yds", "player_pass_tds", "player_pass_completions",
        "player_rush_yds", "player_rush_attempts", "player_receptions",
        "player_reception_yds", "player_rush_reception_yds",
    ],
    "NBA": [
        "player_points", "player_rebounds", "player_assists", "player_threes",
        "player_points_rebounds_assists",
    ],
    "MLB": [
        "batter_hits", "batter_total_bases", "batter_rbis", "batter_runs_scored",
        "batter_home_runs", "pitcher_strikeouts", "pitcher_outs",
    ],
    "NHL": [
        "player_shots_on_goal", "player_points", "player_assists",
        "player_total_saves", "player_goals",
    ],
}

SPORT_ALTERNATE_KEYS = {
    "NFL": [
        "player_pass_yds_alternate", "player_pass_tds_alternate",
        "player_rush_yds_alternate", "player_receptions_alternate",
        "player_reception_yds_alternate", "player_rush_reception_yds_alternate",
    ],
    "NBA": [
        "player_points_alternate", "player_rebounds_alternate",
        "player_assists_alternate", "player_threes_alternate",
        "player_points_rebounds_assists_alternate",
    ],
    "MLB": [
        "batter_hits_alternate", "batter_total_bases_alternate",
        "batter_home_runs_alternate", "batter_rbis_alternate",
        "pitcher_strikeouts_alternate", "pitcher_outs_alternate",
    ],
    "NHL": [
        "player_points_alternate", "player_assists_alternate",
        "player_goals_alternate", "player_shots_on_goal_alternate",
        "player_total_saves_alternate",
    ],
}

EXTRA_GAME_MARKETS = {
    "NFL": ["team_totals"],
    "NBA": ["team_totals", "h2h_h1", "spreads_h1", "totals_h1"],
    "MLB": ["h2h_1st_5_innings", "spreads_1st_5_innings", "totals_1st_5_innings"],
    "NHL": ["h2h_p1", "spreads_p1", "totals_p1"],
}

GENERIC_MARKET_LABELS = {
    "h2h": "Moneyline",
    "spreads": "Spread",
    "totals": "Game total",
    "alternate_spreads": "Alternate spread",
    "alternate_totals": "Alternate total",
    "team_totals": "Team total",
    "alternate_team_totals": "Alternate team total",
    "h2h_h1": "1st half moneyline",
    "spreads_h1": "1st half spread",
    "totals_h1": "1st half total",
    "h2h_1st_5_innings": "First 5 moneyline",
    "spreads_1st_5_innings": "First 5 run line",
    "totals_1st_5_innings": "First 5 total",
    "h2h_p1": "1st period moneyline",
    "spreads_p1": "1st period puck line",
    "totals_p1": "1st period total",
    "outrights": "Championship future",
}
for _sport, _mkts in SPORT_PROP_MARKETS.items():
    GENERIC_MARKET_LABELS.update(_mkts)
for _sport, _keys in SPORT_ALTERNATE_KEYS.items():
    for _key in _keys:
        _base = _key.replace("_alternate", "")
        GENERIC_MARKET_LABELS[_key] = "Alternate " + GENERIC_MARKET_LABELS.get(_base, _base.replace("_", " "))


def market_label(market_key: str) -> str:
    return GENERIC_MARKET_LABELS.get(str(market_key), str(market_key).replace("_", " ").title())


def market_group(market_key: str) -> str:
    """Broad display/ranking family used by Market Radar.

    The scanner deliberately keeps scorer longshots separate from normal yardage /
    counting props. A 25%-30% payout difference on a +3000 first-scorer market can
    look enormous in relative terms while representing only a small absolute
    probability disagreement, so those rows should not crowd out core lines.
    """
    mk = str(market_key or "")
    low = mk.lower()
    if low == "outrights" or "championship" in low:
        return "Futures"
    if low in {
        "player_1st_td", "player_last_td", "player_anytime_td", "player_tds",
        "player_first_basket", "player_goal_scorer_first",
        "player_goal_scorer_last", "player_goal_scorer_anytime",
    } or "scorer" in low:
        return "TD / scorer props"
    if low.startswith(("h2h", "spreads", "totals", "team_totals", "alternate_spreads", "alternate_totals")):
        return "Game lines"
    if low.startswith(("player_", "batter_", "pitcher_")):
        return "Core player props"
    return "Other markets"


def _longshot_gap_score_cap(market_key: str, median_prob: float | None) -> float:
    """Cap ranking inflation from large relative payout gaps on tiny probabilities."""
    mk = str(market_key or "").lower()
    group = market_group(mk)
    if group == "TD / scorer props":
        if any(x in mk for x in ("1st", "first", "last")):
            return 74.0
        return 82.0
    if median_prob is not None:
        if median_prob < 0.05:
            return 74.0
        if median_prob < 0.10:
            return 78.0
        if median_prob < 0.15:
            return 83.0
    return 100.0


def _num(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _best_link(outcome, market, bookmaker):
    return outcome.get("link") or market.get("link") or bookmaker.get("link") or ""


def _bet_limit(outcome):
    # The exact field can vary by provider/market; keep this permissive.
    for key in ("bet_limit", "betLimit", "max_bet", "maxBet", "limit"):
        if key in outcome:
            x = _num(outcome.get(key))
            if x is not None:
                return x
    return None


def normalize_market_payloads(payloads, sport_label: str | None = None) -> pd.DataFrame:
    """Normalize featured, event-prop and outright responses into one offer table."""
    if payloads is None:
        return pd.DataFrame()
    if isinstance(payloads, dict):
        payloads = [payloads]
    rows = []
    for event in payloads or []:
        if not isinstance(event, dict):
            continue
        sport = sport_label or event.get("sport_title") or event.get("sport_key") or ""
        home, away = event.get("home_team"), event.get("away_team")
        matchup = f"{away} @ {home}" if away and home else (event.get("sport_title") or sport or "Futures")
        for b in event.get("bookmakers", []) or []:
            bkey = str(b.get("key") or "")
            bname = str(b.get("title") or bkey)
            for m in b.get("markets", []) or []:
                mkey = str(m.get("key") or "")
                mlabel = market_label(mkey)
                updated = m.get("last_update") or b.get("last_update")
                for o in m.get("outcomes", []) or []:
                    odds = _num(o.get("price"))
                    if odds is None:
                        continue
                    line = _num(o.get("point"))
                    side = str(o.get("name") or "")
                    description = str(o.get("description") or "").strip()
                    if mkey == "outrights":
                        subject = "Championship"
                    elif description:
                        subject = description
                    else:
                        subject = "Game"
                    rows.append({
                        "Sport": sport,
                        "Sport Key": event.get("sport_key"),
                        "Event ID": event.get("id"),
                        "Matchup": matchup,
                        "Commence": event.get("commence_time"),
                        "Home": home,
                        "Away": away,
                        "Market Key": mkey,
                        "Market": mlabel,
                        "Subject": subject,
                        "Side": side,
                        "Line": line,
                        "Odds": odds,
                        "Book Key": bkey,
                        "Book": bname,
                        "Exchange": bkey in EXCHANGE_BOOK_KEYS,
                        "Updated": updated,
                        "Link": _best_link(o, m, b),
                        "Bet Limit": _bet_limit(o),
                    })
    return pd.DataFrame(rows)


def combine_normalized_offer_frames(frames) -> pd.DataFrame:
    good = [x for x in (frames or []) if isinstance(x, pd.DataFrame) and not x.empty]
    return pd.concat(good, ignore_index=True) if good else pd.DataFrame()


def _fmt_line(x):
    x = _num(x)
    if x is None:
        return ""
    return f"{x:g}"


def _fmt_odds_text(x):
    x = _num(x)
    if x is None:
        return ""
    return f"{int(round(x)):+d}"


def _effective_decimal(row, exchange_fee_buffer: float = 0.0):
    d = american_to_decimal(row.get("Odds"))
    if d is None:
        return None
    fee = max(0.0, min(0.25, float(exchange_fee_buffer or 0.0)))
    if bool(row.get("Exchange")) and fee > 0:
        # Conservative generic buffer: haircut only the profit component. Actual fee
        # schedules differ by exchange, so the UI labels this an approximation.
        d = 1.0 + (d - 1.0) * (1.0 - fee)
    return d


def _best_offer_adjusted(rows, exchange_fee_buffer: float = 0.0):
    """Return the row with the best payout after the configured exchange-fee buffer."""
    valid = []
    for r in rows or []:
        d = _effective_decimal(r, exchange_fee_buffer)
        if d is not None:
            valid.append((d, r))
    if not valid:
        return None
    return max(valid, key=lambda x: x[0])[1]


def _best_pair_adjusted(a_rows, b_rows, exchange_fee_buffer: float = 0.0):
    combos = []
    for a in a_rows:
        da = _effective_decimal(a, exchange_fee_buffer)
        if da is None:
            continue
        for b in b_rows:
            if a.get("Book Key") == b.get("Book Key"):
                continue
            db = _effective_decimal(b, exchange_fee_buffer)
            if db is None:
                continue
            combos.append((1.0 / da + 1.0 / db, a, b, da, db))
    if not combos:
        return None
    return min(combos, key=lambda x: x[0])


def _arb_math_effective(a, b, total_stake=100.0, exchange_fee_buffer: float = 0.0):
    da = _effective_decimal(a, exchange_fee_buffer)
    db = _effective_decimal(b, exchange_fee_buffer)
    if da is None or db is None:
        return None
    inv = 1.0 / da + 1.0 / db
    if inv <= 0:
        return None
    stake_a = float(total_stake) * (1.0 / da) / inv
    stake_b = float(total_stake) * (1.0 / db) / inv
    payout = float(total_stake) / inv
    return {
        "Implied Sum": inv,
        "ROI": 1.0 / inv - 1.0,
        "Stake A": stake_a,
        "Stake B": stake_b,
        "Guaranteed Payout": payout,
        "Guaranteed Profit": payout - float(total_stake),
    }


def _arb_edge_label(roi, rows, exchange_fee_buffer: float = 0.0):
    exchange_involved = any(bool(r.get("Exchange")) for r in rows if isinstance(r, dict))
    if exchange_involved and float(exchange_fee_buffer or 0.0) > 0:
        return f"{roi:.2%} fee-buffered arb"
    if exchange_involved:
        return f"{roi:.2%} before exchange fees"
    return f"{roi:.2%} theoretical arb"


def _arb_profit_word(rows, exchange_fee_buffer: float = 0.0):
    exchange_involved = any(bool(r.get("Exchange")) for r in rows if isinstance(r, dict))
    if exchange_involved and float(exchange_fee_buffer or 0.0) > 0:
        return "estimated locked profit after the configured fee buffer"
    if exchange_involved:
        return "estimated locked profit before exchange fees"
    return "theoretical locked profit"


def _selection_text(row):
    subject = str(row.get("Subject") or "")
    side = str(row.get("Side") or "")
    line = _num(row.get("Line"))
    mkey = str(row.get("Market Key") or "")
    if subject and subject not in {"Game", "Championship"}:
        prefix = f"{subject} "
    else:
        prefix = ""
    if line is not None and (side.lower() in {"over", "under"} or mkey.startswith("spreads") or "spread" in mkey):
        if side.lower() in {"over", "under"}:
            return f"{prefix}{side} {line:g}".strip()
        return f"{prefix}{side} {line:+g}".strip()
    return f"{prefix}{side}".strip()


def _opp_row(*, score, kind, row, pick, compare, edge, why, book=None, odds=None, book2=None, odds2=None,
             line_adv=None, payout_adv=None, arb_roi=None, middle_width=None, details=""):
    return {
        "Score": round(float(max(0, min(100, score))), 1),
        "Type": kind,
        "Sport": row.get("Sport", ""),
        "Matchup": row.get("Matchup", ""),
        "Commence": row.get("Commence"),
        "Market": row.get("Market", market_label(row.get("Market Key", ""))),
        "Market Key": row.get("Market Key", ""),
        "Market Group": market_group(row.get("Market Key", "")),
        "Subject": row.get("Subject", ""),
        "Pick": pick,
        "Book": book if book is not None else row.get("Book", ""),
        "Odds": odds if odds is not None else row.get("Odds"),
        "Book 2": book2 or "",
        "Odds 2": odds2,
        "Compare": compare,
        "Edge": edge,
        "Why": why,
        "Event ID": row.get("Event ID"),
        "Line Advantage": line_adv,
        "Payout Advantage": payout_adv,
        "Arb ROI": arb_roi,
        "Middle Width": middle_width,
        "Link": row.get("Link", ""),
        "Bet Limit": row.get("Bet Limit"),
        "Details": details,
    }


def _mainish_offer(rows):
    """Choose a representative/main-ish quote when a feed returns multiple points.

    Pikkit-style boards usually show each book's main line, not every alternate rung.
    Main player-prop/spread/total prices are usually closest to standard two-way juice,
    so prefer prices whose absolute American odds are closest to 110. This avoids
    treating a -350 alternate rung as that book's representative market line.
    """
    valid = []
    for r in rows or []:
        o = _num(r.get("Odds"))
        if o is None or o == 0:
            continue
        # Main two-way prices are generally near even/standard juice. The second
        # term breaks +110/-110 style ties toward the price closer to 50% implied.
        p = implied_prob(o)
        score = (abs(abs(o) - 110.0), abs((p if p is not None else 0.5) - 0.5))
        valid.append((score, r))
    return min(valid, key=lambda x: x[0])[1] if valid else None


def price_outlier_opportunities(offers: pd.DataFrame, min_payout_advantage: float = 0.015,
                                min_books: int = 2, exchange_fee_buffer: float = 0.0) -> pd.DataFrame:
    """Find exact-line price anomalies without mixing sportsbook and exchange consensus.

    Important rules:
    - alternate ladders are excluded from *price-consensus* alerts (they remain usable
      for arbs/middles), because heavy juice on an alternate point is normal;
    - sportsbook outliers are compared only with other sportsbooks;
    - exchange gaps compare the best exchange quote with the sportsbook median;
    - an exchange merely being present no longer turns a sportsbook alert into an
      ``EXCHANGE GAP``.
    """
    if offers is None or offers.empty:
        return pd.DataFrame()
    req = {"Event ID", "Market Key", "Subject", "Side", "Line", "Odds", "Book Key", "Book"}
    if not req.issubset(offers.columns):
        return pd.DataFrame()

    base = offers[~offers["Market Key"].astype(str).str.contains("alternate", case=False, na=False)].copy()
    rows = []
    keys = ["Event ID", "Market Key", "Subject", "Side", "Line"]

    for _, g in base.groupby(keys, dropna=False):
        book_rows = []
        for _, bg in g.groupby("Book Key", dropna=False):
            rr = _best_offer_adjusted(bg.to_dict("records"), exchange_fee_buffer)
            if rr:
                book_rows.append(rr)
        if len(book_rows) < 2:
            continue

        sports = [r for r in book_rows if not bool(r.get("Exchange"))]
        exchanges = [r for r in book_rows if bool(r.get("Exchange"))]

        # 1) Sportsbook-vs-sportsbook exact-line price outlier.
        if len(sports) >= max(2, int(min_books)):
            candidate = _best_offer_adjusted(sports, exchange_fee_buffer)
            others = [r for r in sports if r.get("Book Key") != candidate.get("Book Key")]
            cand_dec = _effective_decimal(candidate, exchange_fee_buffer)
            other_dec = [_effective_decimal(r, exchange_fee_buffer) for r in others]
            other_dec = [x for x in other_dec if x]
            if cand_dec is not None and other_dec:
                median_dec = float(pd.Series(other_dec).median())
                advantage = cand_dec / median_dec - 1.0
                if advantage >= float(min_payout_advantage):
                    median_prob = 1.0 / median_dec
                    median_american = fair_american(median_prob)
                    score = 45 + min(40, advantage * 330) + min(7, max(0, len(sports) - 2) * 1.5)
                    score = min(score, _longshot_gap_score_cap(candidate.get("Market Key"), median_prob))
                    rows.append(_opp_row(
                        score=score,
                        kind="PRICE OUTLIER",
                        row=candidate,
                        pick=_selection_text(candidate),
                        compare=f"Same-line sportsbook median {_fmt_odds_text(median_american)}",
                        edge=f"{advantage:+.1%} payout",
                        why=(f"{candidate.get('Book')} is paying {_fmt_odds_text(candidate.get('Odds'))} "
                             f"for the exact same line; the other sportsbook median at this exact point is "
                             f"about {_fmt_odds_text(median_american)}."),
                        payout_adv=advantage,
                        details=f"Exact-line comparison across {len(sports)} sportsbooks. Exchange quotes are not used to set this sportsbook consensus.",
                    ))

        # 2) Exchange-vs-sportsbook exact-line gap.
        # Require at least one sportsbook comparator; two is preferable and gets a higher score.
        if exchanges and sports:
            ex = _best_offer_adjusted(exchanges, exchange_fee_buffer)
            ex_dec = _effective_decimal(ex, exchange_fee_buffer)
            sp_dec = [_effective_decimal(r, exchange_fee_buffer) for r in sports]
            sp_dec = [x for x in sp_dec if x]
            if ex_dec is not None and sp_dec:
                median_dec = float(pd.Series(sp_dec).median())
                advantage = ex_dec / median_dec - 1.0
                if advantage >= float(min_payout_advantage):
                    median_prob = 1.0 / median_dec
                    median_american = fair_american(median_prob)
                    score = 49 + min(40, advantage * 330) + min(7, max(0, len(sports) - 1) * 1.5)
                    score = min(score, _longshot_gap_score_cap(ex.get("Market Key"), median_prob))
                    rows.append(_opp_row(
                        score=score,
                        kind="EXCHANGE GAP",
                        row=ex,
                        pick=_selection_text(ex),
                        compare=f"Same-line sportsbook median {_fmt_odds_text(median_american)}",
                        edge=f"{advantage:+.1%} payout",
                        why=(f"{ex.get('Book')} is paying {_fmt_odds_text(ex.get('Odds'))} at this exact line, "
                             f"versus a sportsbook median of about {_fmt_odds_text(median_american)}."),
                        payout_adv=advantage,
                        details=f"Exchange quote compared with {len(sports)} sportsbook quote(s) at the exact same point. Fee buffer is applied to the exchange quote when configured.",
                    ))

    return pd.DataFrame(rows)


def _line_direction(market_key: str, side: str):
    s = str(side).lower()
    mk = str(market_key)
    if s == "over":
        return -1.0  # lower line is better
    if s == "under":
        return 1.0   # higher line is better
    if "spread" in mk:
        return 1.0   # more points is better for the selected team
    return None


def line_outlier_opportunities(offers: pd.DataFrame, min_line_advantage: float = 0.5,
                               min_books: int = 2) -> pd.DataFrame:
    """Compare representative *main* lines, not the most extreme alternate rung.

    This is the Pikkit-like comparison: one representative line per book, then surface
    a sportsbook or exchange whose line is materially better than sportsbook consensus.
    """
    if offers is None or offers.empty:
        return pd.DataFrame()
    rows = []
    gkeys = ["Event ID", "Market Key", "Subject", "Side"]
    line_df = offers[offers["Line"].notna()].copy()
    line_df = line_df[~line_df["Market Key"].astype(str).str.contains("alternate", case=False, na=False)]

    for _, g in line_df.groupby(gkeys, dropna=False):
        direction = _line_direction(g.iloc[0]["Market Key"], g.iloc[0]["Side"])
        if direction is None:
            continue

        reps = []
        for _, bg in g.groupby("Book Key", dropna=False):
            rr = _mainish_offer(bg.to_dict("records"))
            if rr:
                reps.append(rr)
        if len(reps) < 2:
            continue

        sports = [r for r in reps if not bool(r.get("Exchange"))]
        exchanges = [r for r in reps if bool(r.get("Exchange"))]

        # Sportsbook line outlier versus other sportsbook main lines.
        if len(sports) >= max(2, int(min_books)):
            best_metric = max(direction * float(r["Line"]) for r in sports)
            tied = [r for r in sports if abs(direction * float(r["Line"]) - best_metric) < 1e-9]
            candidate = _best_offer_adjusted(tied, 0.0) or tied[0]
            others = [r for r in sports if r.get("Book Key") != candidate.get("Book Key")]
            other_lines = [float(r["Line"]) for r in others]
            if other_lines:
                median_line = float(pd.Series(other_lines).median())
                advantage = direction * (float(candidate["Line"]) - median_line)
                if advantage >= float(min_line_advantage) - 1e-9:
                    rel = advantage / max(1.0, abs(median_line))
                    score = 56 + min(28, rel * 320) + min(7, max(0, len(sports) - 2) * 1.4)
                    rows.append(_opp_row(
                        score=score,
                        kind="LINE OUTLIER",
                        row=candidate,
                        pick=_selection_text(candidate),
                        compare=f"Sportsbook main-line median {median_line:g}",
                        edge=f"{advantage:g} better line",
                        why=(f"{candidate.get('Book')} has {_selection_text(candidate)} at roughly main-line juice; "
                             f"the other sportsbooks center around {median_line:g} for the same side."),
                        line_adv=advantage,
                        details=f"Representative main-line comparison across {len(sports)} sportsbooks; alternate ladders are excluded.",
                    ))

        # Exchange line gap versus sportsbook main-line consensus.
        if exchanges and sports:
            best_metric = max(direction * float(r["Line"]) for r in exchanges)
            tied = [r for r in exchanges if abs(direction * float(r["Line"]) - best_metric) < 1e-9]
            candidate = _best_offer_adjusted(tied, 0.0) or tied[0]
            median_line = float(pd.Series([float(r["Line"]) for r in sports]).median())
            advantage = direction * (float(candidate["Line"]) - median_line)
            if advantage >= float(min_line_advantage) - 1e-9:
                rel = advantage / max(1.0, abs(median_line))
                score = 54 + min(28, rel * 320) + min(7, max(0, len(sports) - 1) * 1.4)
                rows.append(_opp_row(
                    score=score,
                    kind="EXCHANGE GAP",
                    row=candidate,
                    pick=_selection_text(candidate),
                    compare=f"Sportsbook main-line median {median_line:g}",
                    edge=f"{advantage:g} better line",
                    why=(f"{candidate.get('Book')} has {_selection_text(candidate)}; sportsbook main lines center around "
                         f"{median_line:g} for the same side."),
                    line_adv=advantage,
                    details=f"Exchange main-line quote compared with {len(sports)} sportsbook main-line quote(s); alternate ladders are excluded.",
                ))

    return pd.DataFrame(rows)

def _best_two_side_pair(g: pd.DataFrame, side_a: str, side_b: str):
    a = g[g["Side"].astype(str).str.lower().eq(str(side_a).lower())].to_dict("records")
    b = g[g["Side"].astype(str).str.lower().eq(str(side_b).lower())].to_dict("records")
    return _distinct_best_pair(a, b)


def exact_arb_opportunities(offers: pd.DataFrame, total_stake: float = 100.0, exchange_fee_buffer: float = 0.0) -> pd.DataFrame:
    """Find two-way arbs at the same line plus exact two-team moneylines."""
    if offers is None or offers.empty:
        return pd.DataFrame()
    rows = []

    # Over/Under and Yes/No exact-line arbs.
    keys = ["Event ID", "Market Key", "Subject", "Line"]
    for _, g in offers.groupby(keys, dropna=False):
        side_l = {str(x).lower() for x in g["Side"].dropna().unique()}
        pair_names = None
        if {"over", "under"}.issubset(side_l):
            pair_names = ("Over", "Under")
        elif {"yes", "no"}.issubset(side_l):
            pair_names = ("Yes", "No")
        if not pair_names:
            continue
        a_rows = g[g["Side"].astype(str).str.lower().eq(pair_names[0].lower())].to_dict("records")
        b_rows = g[g["Side"].astype(str).str.lower().eq(pair_names[1].lower())].to_dict("records")
        pair = _best_pair_adjusted(a_rows, b_rows, exchange_fee_buffer)
        if not pair or pair[0] >= 1.0:
            continue
        inv, a, b, _, _ = pair
        calc = _arb_math_effective(a, b, total_stake, exchange_fee_buffer)
        if not calc:
            continue
        roi = calc["ROI"]
        score = 86 + min(14, roi * 900)
        pa, pb = _selection_text(a), _selection_text(b)
        edge = _arb_edge_label(roi, [a, b], exchange_fee_buffer)
        why = f"{a['Book']} {pa} {_fmt_odds_text(a['Odds'])} + {b['Book']} {pb} {_fmt_odds_text(b['Odds'])} produce an implied sum below 100% after the configured exchange-fee buffer, when applicable."
        details = f"${calc['Stake A']:.2f} on {pa}; ${calc['Stake B']:.2f} on {pb}; {_arb_profit_word([a,b], exchange_fee_buffer)} ≈ ${calc['Guaranteed Profit']:.2f} per ${float(total_stake):.2f}."
        rows.append(_opp_row(
            score=score, kind="TRUE ARB", row=a, pick=f"{pa} / {pb}",
            compare=f"{a['Book']} ↔ {b['Book']}", edge=edge, why=why,
            book=a["Book"], odds=a["Odds"], book2=b["Book"], odds2=b["Odds"], arb_roi=roi, details=details,
        ))

    # Two-team moneyline arbs (including period/inning 2-way h2h markets).
    h = offers[offers["Market Key"].astype(str).str.startswith("h2h")].copy()
    h = h[~h["Market Key"].astype(str).str.contains("3_way", na=False)]
    for _, g in h.groupby(["Event ID", "Market Key"], dropna=False):
        sides = [x for x in g["Side"].dropna().unique() if str(x).strip()]
        if len(sides) != 2:
            continue
        a_rows = g[g["Side"].eq(sides[0])].to_dict("records")
        b_rows = g[g["Side"].eq(sides[1])].to_dict("records")
        pair = _best_pair_adjusted(a_rows, b_rows, exchange_fee_buffer)
        if not pair or pair[0] >= 1.0:
            continue
        inv, a, b, _, _ = pair
        calc = _arb_math_effective(a, b, total_stake, exchange_fee_buffer)
        if not calc:
            continue
        roi = calc["ROI"]
        score = 86 + min(14, roi * 900)
        why = f"Best opposing prices are {a['Book']} {a['Side']} {_fmt_odds_text(a['Odds'])} and {b['Book']} {b['Side']} {_fmt_odds_text(b['Odds'])}."
        details = f"${calc['Stake A']:.2f} on {a['Side']}; ${calc['Stake B']:.2f} on {b['Side']}; {_arb_profit_word([a,b], exchange_fee_buffer)} ≈ ${calc['Guaranteed Profit']:.2f} per ${float(total_stake):.2f}."
        rows.append(_opp_row(
            score=score, kind="TRUE ARB", row=a, pick=f"{a['Side']} / {b['Side']}",
            compare=f"{a['Book']} ↔ {b['Book']}", edge=_arb_edge_label(roi, [a,b], exchange_fee_buffer), why=why,
            book=a["Book"], odds=a["Odds"], book2=b["Book"], odds2=b["Odds"], arb_roi=roi, details=details,
        ))

    # Exact spread arbs: opposite teams at inverse lines.
    s = offers[offers["Market Key"].astype(str).str.contains("spreads", na=False)].copy()
    for (_, mk), g in s.groupby(["Event ID", "Market Key"], dropna=False):
        sides = [x for x in g["Side"].dropna().unique() if str(x).strip()]
        if len(sides) != 2:
            continue
        side_a, side_b = sides[0], sides[1]
        for line in sorted(set(g[g["Side"].eq(side_a)]["Line"].dropna())):
            ga = g[(g["Side"].eq(side_a)) & (g["Line"] == line)]
            gb = g[(g["Side"].eq(side_b)) & (g["Line"] == -float(line))]
            if ga.empty or gb.empty:
                continue
            pair = _best_pair_adjusted(ga.to_dict("records"), gb.to_dict("records"), exchange_fee_buffer)
            if not pair or pair[0] >= 1.0:
                continue
            inv, a, b, _, _ = pair
            calc = _arb_math_effective(a, b, total_stake, exchange_fee_buffer)
            if not calc:
                continue
            roi = calc["ROI"]
            score = 86 + min(14, roi * 900)
            pa, pb = _selection_text(a), _selection_text(b)
            rows.append(_opp_row(
                score=score, kind="TRUE ARB", row=a, pick=f"{pa} / {pb}",
                compare=f"{a['Book']} ↔ {b['Book']}", edge=_arb_edge_label(roi, [a,b], exchange_fee_buffer),
                why=f"Opposite spread sides at the same effective line create an arb after the configured exchange-fee buffer, when applicable.",
                book=a["Book"], odds=a["Odds"], book2=b["Book"], odds2=b["Odds"], arb_roi=roi,
                details=f"${calc['Stake A']:.2f} on {pa}; ${calc['Stake B']:.2f} on {pb}; {_arb_profit_word([a,b], exchange_fee_buffer)} ≈ ${calc['Guaranteed Profit']:.2f} per ${float(total_stake):.2f}.",
            ))
    return pd.DataFrame(rows)


def middle_opportunities(offers: pd.DataFrame, total_stake: float = 100.0, exchange_fee_buffer: float = 0.0) -> pd.DataFrame:
    """Find favorable line gaps. A price-qualified middle can also be a true arb."""
    if offers is None or offers.empty:
        return pd.DataFrame()
    rows = []
    base = offers[~offers["Market Key"].astype(str).str.contains("alternate", case=False, na=False)].copy()

    # Over/Under middles for totals, team totals and player props.
    for _, g in base.groupby(["Event ID", "Market Key", "Subject"], dropna=False):
        ov = g[g["Side"].astype(str).str.lower().eq("over") & g["Line"].notna()]
        un = g[g["Side"].astype(str).str.lower().eq("under") & g["Line"].notna()]
        if ov.empty or un.empty:
            continue
        lo, hi = float(ov["Line"].min()), float(un["Line"].max())
        if lo >= hi:
            continue
        a = _best_offer_adjusted(ov[ov["Line"] == lo].to_dict("records"), exchange_fee_buffer)
        b = _best_offer_adjusted(un[un["Line"] == hi].to_dict("records"), exchange_fee_buffer)
        if not a or not b or a.get("Book Key") == b.get("Book Key"):
            continue
        width = hi - lo
        da, db = _effective_decimal(a, exchange_fee_buffer), _effective_decimal(b, exchange_fee_buffer)
        inv = (1.0 / da + 1.0 / db) if da and db else None
        arb = inv is not None and inv < 1.0
        roi = (1.0 / inv - 1.0) if arb else None
        mid_rel = width / max(1.0, abs((lo + hi) / 2.0))
        score = 64 + min(24, mid_rel * 380) + (10 if arb else 0)
        pa, pb = _selection_text(a), _selection_text(b)
        kind = "ARB + MIDDLE" if arb else "MIDDLE"
        edge = f"{width:g} line window" + (f" · {_arb_edge_label(roi, [a,b], exchange_fee_buffer)}" if arb else "")
        details = ""
        if arb:
            calc = _arb_math_effective(a, b, total_stake, exchange_fee_buffer)
            if calc:
                details = f"Outside the middle this prices as an arb after the configured fee buffer when applicable: ${calc['Stake A']:.2f} / ${calc['Stake B']:.2f}; ≈ ${calc['Guaranteed Profit']:.2f} profit per ${float(total_stake):.2f}. Inside the middle both bets can win."
        rows.append(_opp_row(
            score=score, kind=kind, row=a, pick=f"{pa} + {pb}",
            compare=f"{a['Book']} ↔ {b['Book']}", edge=edge,
            why=f"One book offers {pa} while another offers {pb}; the {lo:g}–{hi:g} gap can make both sides win in the middle.",
            book=a["Book"], odds=a["Odds"], book2=b["Book"], odds2=b["Odds"],
            arb_roi=roi, middle_width=width, details=details,
        ))

    # Spread middles.
    spreads = base[base["Market Key"].astype(str).str.contains("spreads", na=False)].copy()
    for _, g in spreads.groupby(["Event ID", "Market Key"], dropna=False):
        sides = [x for x in g["Side"].dropna().unique() if str(x).strip()]
        if len(sides) != 2:
            continue
        ga = g[g["Side"].eq(sides[0]) & g["Line"].notna()]
        gb = g[g["Side"].eq(sides[1]) & g["Line"].notna()]
        if ga.empty or gb.empty:
            continue
        la, lb = float(ga["Line"].max()), float(gb["Line"].max())
        width = la + lb
        if width <= 0:
            continue
        a = _best_offer_adjusted(ga[ga["Line"] == la].to_dict("records"), exchange_fee_buffer)
        b = _best_offer_adjusted(gb[gb["Line"] == lb].to_dict("records"), exchange_fee_buffer)
        if not a or not b or a.get("Book Key") == b.get("Book Key"):
            continue
        da, db = _effective_decimal(a, exchange_fee_buffer), _effective_decimal(b, exchange_fee_buffer)
        inv = (1.0 / da + 1.0 / db) if da and db else None
        arb = inv is not None and inv < 1.0
        roi = (1.0 / inv - 1.0) if arb else None
        score = 64 + min(24, width * 5.0) + (10 if arb else 0)
        kind = "ARB + MIDDLE" if arb else "MIDDLE"
        pa, pb = _selection_text(a), _selection_text(b)
        details = ""
        if arb:
            calc = _arb_math_effective(a, b, total_stake, exchange_fee_buffer)
            if calc:
                details = f"Arb component after the configured fee buffer when applicable: ${calc['Stake A']:.2f} / ${calc['Stake B']:.2f}; ≈ ${calc['Guaranteed Profit']:.2f} profit per ${float(total_stake):.2f}."
        rows.append(_opp_row(
            score=score, kind=kind, row=a, pick=f"{pa} + {pb}",
            compare=f"{a['Book']} ↔ {b['Book']}",
            edge=f"{width:g}-point overlap" + (f" · {_arb_edge_label(roi, [a,b], exchange_fee_buffer)}" if arb else ""),
            why=f"The two best spread numbers overlap by {width:g} points.",
            book=a["Book"], odds=a["Odds"], book2=b["Book"], odds2=b["Odds"],
            arb_roi=roi, middle_width=width, details=details,
        ))
    return pd.DataFrame(rows)


def future_arb_opportunities(offers: pd.DataFrame, total_stake: float = 100.0, exchange_fee_buffer: float = 0.0) -> pd.DataFrame:
    """Multi-outcome arb check for championship futures."""
    if offers is None or offers.empty:
        return pd.DataFrame()
    f = offers[offers["Market Key"].eq("outrights")].copy()
    if f.empty:
        return pd.DataFrame()
    rows = []
    for _, g in f.groupby(["Sport", "Event ID", "Market Key"], dropna=False):
        bests = []
        for outcome, og in g.groupby("Side", dropna=False):
            rr = _best_offer_adjusted(og.to_dict("records"), exchange_fee_buffer)
            if rr:
                bests.append(rr)
        if len(bests) < 2:
            continue
        inv = sum(1.0 / _effective_decimal(r, exchange_fee_buffer) for r in bests if _effective_decimal(r, exchange_fee_buffer))
        if inv >= 1.0:
            continue
        roi = 1.0 / inv - 1.0
        payout = float(total_stake) / inv
        stakes = []
        for r in bests:
            d = _effective_decimal(r, exchange_fee_buffer)
            stake = float(total_stake) * (1.0 / d) / inv
            stakes.append(f"{r['Side']} {r['Book']} {_fmt_odds_text(r['Odds'])}: ${stake:.2f}")
        first = bests[0]
        rows.append(_opp_row(
            score=88 + min(12, roi * 700), kind="TRUE ARB (FUTURE)", row=first,
            pick=f"Cover all {len(bests)} outcomes", compare="Best price on each outcome",
            edge=_arb_edge_label(roi, bests, exchange_fee_buffer),
            why=f"The best available prices across all {len(bests)} listed outcomes sum to less than 100% implied probability after the configured exchange-fee buffer, when applicable.",
            book="Multiple books", odds=None, arb_roi=roi,
            details=" | ".join(stakes) + f" | Estimated locked payout ≈ ${payout:.2f} on ${float(total_stake):.2f} total stake under the configured fee assumptions.",
        ))
    return pd.DataFrame(rows)


def build_opportunity_board(offers: pd.DataFrame, *, total_stake: float = 100.0,
                            min_payout_advantage: float = 0.015,
                            min_line_advantage: float = 0.5,
                            min_books: int = 2,
                            include_futures_arb: bool = True,
                            exchange_fee_buffer: float = 0.0) -> pd.DataFrame:
    """One ranked board: arbs, middles, line outliers, price outliers and exchange gaps."""
    if offers is None or offers.empty:
        return pd.DataFrame()
    frames = [
        exact_arb_opportunities(offers, total_stake=total_stake, exchange_fee_buffer=exchange_fee_buffer),
        middle_opportunities(offers, total_stake=total_stake, exchange_fee_buffer=exchange_fee_buffer),
        line_outlier_opportunities(offers, min_line_advantage=min_line_advantage, min_books=min_books),
        price_outlier_opportunities(offers, min_payout_advantage=min_payout_advantage, min_books=min_books, exchange_fee_buffer=exchange_fee_buffer),
    ]
    if include_futures_arb:
        frames.append(future_arb_opportunities(offers, total_stake=total_stake, exchange_fee_buffer=exchange_fee_buffer))
    frames = [x for x in frames if isinstance(x, pd.DataFrame) and not x.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "Market Group" not in out.columns:
        out["Market Group"] = out.get("Market Key", "").map(market_group)
    else:
        missing_group = out["Market Group"].isna() | out["Market Group"].astype(str).eq("")
        if missing_group.any():
            out.loc[missing_group, "Market Group"] = out.loc[missing_group, "Market Key"].map(market_group)
    # Remove exact duplicate detector outputs while preserving different opportunity types.
    dedupe = ["Type", "Sport", "Event ID", "Market Key", "Subject", "Pick", "Book", "Book 2"]
    present = [c for c in dedupe if c in out.columns]
    out = out.sort_values(["Score", "Arb ROI", "Payout Advantage", "Line Advantage"],
                          ascending=[False, False, False, False], na_position="last")
    out = out.drop_duplicates(present, keep="first").reset_index(drop=True)
    return out


def apply_watchlist(opportunities: pd.DataFrame, terms) -> pd.DataFrame:
    if opportunities is None or opportunities.empty:
        return opportunities
    terms = [str(x).strip().lower() for x in (terms or []) if str(x).strip()]
    out = opportunities.copy()
    if not terms:
        out["Watch"] = False
        return out
    cols = [c for c in ["Sport", "Matchup", "Market", "Subject", "Pick", "Book", "Book 2", "Why"] if c in out.columns]
    def hit(row):
        text = " ".join(str(row.get(c, "")) for c in cols).lower()
        for term in terms:
            tokens = [t for t in term.replace("+", " ").split() if len(t) >= 2]
            if tokens and all(t in text for t in tokens):
                return True
        return False
    out["Watch"] = out.apply(hit, axis=1)
    out = out.sort_values(["Watch", "Score"], ascending=[False, False]).reset_index(drop=True)
    return out


def book_coverage_summary(offers: pd.DataFrame, selected_book_keys=None) -> pd.DataFrame:
    if offers is None or offers.empty:
        return pd.DataFrame()
    g = offers.groupby(["Book Key", "Book"], dropna=False).agg(
        Offers=("Odds", "size"),
        Events=("Event ID", "nunique"),
        Markets=("Market Key", "nunique"),
        Latest=("Updated", "max"),
    ).reset_index()
    if selected_book_keys:
        wanted = pd.DataFrame({"Book Key": list(selected_book_keys)})
        g = wanted.merge(g, on="Book Key", how="left")
        g["Offers"] = g["Offers"].fillna(0).astype(int)
        g["Events"] = g["Events"].fillna(0).astype(int)
        g["Markets"] = g["Markets"].fillna(0).astype(int)
        g["Book"] = g["Book"].fillna(g["Book Key"])
    return g.sort_values(["Offers", "Book"], ascending=[False, True]).reset_index(drop=True)
