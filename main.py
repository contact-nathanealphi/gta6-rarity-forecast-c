#!/usr/bin/env python3
"""Refresh-aware empirical forecast of GTA VI trophy rarities on Exophase.

Idea in one paragraph: Exophase does not update continuously. Between
refreshes the tracked-player count is frozen and every rate sits still
(launch night proved it: 638 players, rates unchanged for four-plus hours).
At a refresh the tracked population jumps and every rate is dragged toward
the rate among the newly tracked players, who are on average much earlier
in the game, so rates fall in a step. Between refreshes the old cohort
keeps playing and its earners creep upward. This program simulates exactly
that process forward from the case checkpoint: a sequence of dilution
steps (each refresh) interleaved with smooth progression (each gap), then
reads five quantiles off five scenarios that span the plausible range of
refresh counts, new-player rates and progression speeds.

Everything the program knows comes from the case file it is given, plus
small per-trophy-class priors (how fast each kind of trophy is earned,
how high its rate can plausibly saturate, how much of it a fresh player
holds). When the case history itself contains refresh jumps, the jump
sizes, the spacing between jumps and the implied new-player rates are
measured from those jumps and override the priors.

Standard library only. No network, no randomness, no model calls.
Two runs on the same case produce byte-identical output.
"""

import glob
import json
import math
import os
import re
from datetime import datetime, timezone

TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z_SCORES = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)

# ---------------------------------------------------------------------------
# Per-class priors.
#
# Each entry: (relative hourly growth of the earner count, long-run
# saturation cap as a fraction, new-player rate as a multiple of the
# trophy's current rate, per-refresh growth of the new-player rate).
#
# growth: between refreshes, the earner count grows RELATIVELY,
# E *= exp(growth * hours). Relative, not toward N: the players who earn
# a trophy next are the ones already playing, not the fresh joiners who
# have barely started -- the launch-night history is the calibration
# (the tracked cohort played four-plus hours with NO rate moving at
# all), so growth is small on a wall-clock scale. A chapter-1 growth of
# 0.02/h takes the original cohort's holders from ~8 to ~200 over a
# week, the day-7 anchor. cap: the share of players who could ever hold
# the trophy in week one (early chapters high, platinum near zero).
# new-player multiple: a player who only just got tracked holds far
# less of a late trophy than the launch-night cohort did -- the live
# 11:27Z snapshot shows the median trophy diluted to ~0.2x of launch
# night, so these multiples sit well below the 0.44x of the synthetic
# fixture. growth (alpha): as the week goes on, the players joining at
# each refresh have had more time to play, so their rates creep up --
# fast for early-story trophies, slow for rare ones.
# ---------------------------------------------------------------------------
PROFILES = {
    # The first trophy everybody earns within hours of starting.
    "first":      (0.05,  0.999, 0.99, 0.00),
    # Story chapters, in order. Earlier chapters are reached sooner.
    "story1":     (0.03,  0.55,  0.20, 0.30),
    "story2":     (0.024, 0.45,  0.20, 0.25),
    "story3":     (0.018, 0.35,  0.16, 0.20),
    "story4":     (0.014, 0.30,  0.16, 0.15),
    # The final mission: reachable only after the whole story.
    "final":      (0.008, 0.20,  0.12, 0.12),
    # Side activities (fishing, arcade, driving, exploring): unlocked
    # early, earned whenever a player feels like it.
    "side":       (0.018, 0.40,  0.24, 0.25),
    # Grindy / completionist trophies: many repetitions or rare feats.
    "grind":      (0.006, 0.15,  0.12, 0.08),
    # Attain 100% completion.
    "completion": (0.0015, 0.02, 0.04, 0.05),
    # The platinum: every other trophy first.
    "platinum":   (0.0008, 0.01, 0.04, 0.05),
}

GRIND_WORDS = (
    "250", "gold medal", "Win 20", "every location", "100 vehicles",
    "50 Challenges", "100 locations", "8 mission strands",
    "Collectible Card", "every spot", "5 Bank Vaults", "all of Leonida",
)

# Priors used when the case history contains no refresh jump at all.
PRIOR_REFRESH_INTERVAL_HOURS = 5.0   # Exophase refreshed about every
                                     # 4-5 h on launch night.
PRIOR_FIRST_JUMP_RATIO = 6.0         # launch-night jump: 638 -> 3,868.
PRIOR_JUMP_RATIO_DECAY = 0.6         # later jumps are smaller in ratio.
PRIOR_JUMP_RATIO_FLOOR = 1.15
# The tracked population cannot grow without bound in a week: GTA V
# reached ~783k tracked PS4 players on Exophase over its whole
# lifetime, so week-one GTA VI is capped here at one million. Without
# the cap, compounding the early 6x jumps over a 4.5-day horizon would
# project tens of millions of tracked players and dilute every rate
# to nothing.
N_CAP = 1000000.0

# How the five scenarios span the uncertainty. Each scenario is a
# (horizon shift as a fraction of one refresh interval, new-player-rate
# multiplier, growth multiplier). The pessimistic end assumes a refresh
# a little sooner than the central case and weaker new players; the
# optimistic end assumes a refresh a little later and faster
# progression. Shifts are half an interval -- a whole extra refresh is
# too coarse a step for a short horizon, and for a long horizon the
# refresh count barely moves, so the rate/growth factors carry the
# spread there.
SCENARIOS = (
    (0.5, 0.6, 0.6),     # q05: refresh sooner, slow new players
    (0.0, 0.8, 0.8),     # q25
    (0.0, 1.0, 1.0),     # q50: central
    (0.0, 1.3, 1.5),     # q75
    (-0.5, 1.7, 2.0),    # q95: refresh later, fast new players
)

# Extra logit spread on top of the scenario spread, per sqrt(day) of
# horizon -- the part of the future no refresh count can capture.
SPREAD_PER_SQRT_DAY = 0.15


def parse_time(text):
    """Parse an ISO-8601 timestamp (with a trailing Z) to UTC datetime."""
    if text is None:
        return None
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None


def clamp(value, low, high):
    return low if value < low else high if value > high else value


def logit(p):
    p = clamp(p, 0.0001, 0.9999)
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return None
    if n % 2:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def classify(trophy, last_rate):
    """Assign a trophy to a behaviour class from its own description.

    Only the case file is consulted: tier, description text and the
    last observed rate. A rate at (or essentially at) 100% can only be
    the opening trophy -- everybody earns it first.
    """
    tier = (trophy.get("tier") or "").lower()
    desc = trophy.get("description") or ""
    if last_rate is not None and last_rate >= 99.5:
        return "first"
    if tier == "platinum":
        return "platinum"
    if "100%" in desc:
        return "completion"
    match = re.search(r"Chapter (\d)", desc)
    if match:
        chapter = int(match.group(1))
        return "story%d" % clamp(chapter, 1, 4)
    if desc.strip() == "Complete the final mission.":
        return "final"
    if any(word in desc for word in GRIND_WORDS):
        return "grind"
    return "side"


def read_history(case):
    """Normalise the case history into (time, players, [rates]) rows.

    Null rates are kept as None and skipped later; a missing
    players_tracked carries the previous read's value forward.
    """
    rows = []
    last_players = None
    for read in case.get("history") or []:
        when = parse_time(read.get("at") or read.get("slot"))
        if when is None:
            continue
        players = read.get("players_tracked")
        if players is None:
            players = last_players
        else:
            last_players = players
        rates = read.get("rates") or []
        rows.append((when, players, rates))
    return rows


def last_rates(rows, n_trophies):
    """Last non-null rate per trophy, walking the history backwards."""
    out = [None] * n_trophies
    for _when, _players, rates in reversed(rows):
        for i in range(min(n_trophies, len(rates))):
            if out[i] is None and rates[i] is not None:
                out[i] = float(rates[i])
        if all(v is not None for v in out):
            break
    return out


def measure_jumps(rows, n_trophies):
    """Measure refresh jumps from the history.

    A jump is a read whose tracked-player count grew. For each jump and
    each trophy, the implied rate among the newly tracked players is
    recovered from the dilution arithmetic:

        r_new = (r2 * N2 - r1 * N1) / (N2 - N1)

    Returns (jump_times, ratios, per-trophy new-player rates from the
    most recent jump, time of the most recent jump).
    """
    jump_times = []
    ratios = []
    new_rates = None
    last_jump_time = None
    for (t1, n1, r1), (t2, n2, r2) in zip(rows, rows[1:]):
        if n1 is None or n2 is None or n2 <= n1:
            continue
        ratio = n2 / float(n1)
        if ratio < 1.02:
            continue
        jump_times.append(t2)
        ratios.append(ratio)
        last_jump_time = t2
        fresh = []
        for i in range(n_trophies):
            a = r1[i] if i < len(r1) else None
            b = r2[i] if i < len(r2) else None
            if a is None or b is None:
                fresh.append(None)
            else:
                implied = (b * n2 - a * n1) / float(n2 - n1)
                fresh.append(clamp(implied, 0.0, 100.0))
        new_rates = fresh
    return jump_times, ratios, new_rates, last_jump_time


def observed_growth(rows, n_trophies):
    """Per-trophy earner growth measured on flat stretches of history.

    While the player count is frozen, any rise in a rate is pure
    progression of the tracked cohort. With N fixed, the log-rate slope
    is the relative growth rate of the earner count: d(ln E)/dt.
    Returns None for trophies with no measurable rise.
    """
    growths = [None] * n_trophies
    for i in range(n_trophies):
        slopes = []
        for (t1, n1, r1), (t2, n2, r2) in zip(rows, rows[1:]):
            if n1 is None or n1 != n2:
                continue
            a = r1[i] if i < len(r1) else None
            b = r2[i] if i < len(r2) else None
            if a is None or b is None or a <= 0.0 or b <= a:
                continue
            hours = (t2 - t1).total_seconds() / 3600.0
            if hours <= 0:
                continue
            slopes.append(math.log(b / a) / hours)
        if slopes:
            growths[i] = median(slopes)
    return growths


def simulate(rate_pct, players, profile, new_base_pct, growth,
             horizon_hours, since_last_hours, interval_hours,
             first_ratio, horizon_shift_hours, new_mult, growth_mult):
    """Run the refresh-step process forward and return the final rate.

    rate_pct / players describe the last read. horizon_shift_hours
    moves the target earlier or later by a fraction of one refresh
    interval (used by the outer scenarios to shift a refresh in or out
    of the window).
    """
    horizon = horizon_hours + (horizon_shift_hours or 0.0)
    if horizon <= 0.0:
        return clamp(rate_pct, 0.01, 99.99)
    growth_h, cap, _new_mult, alpha = profile
    growth = growth * growth_mult
    cap_pct = cap * 100.0
    n = float(players) if players else 1000.0
    earners = clamp(rate_pct, 0.0, 100.0) / 100.0 * n

    def progress(earners, n, dt):
        # Between refreshes the earners grow relatively (the players
        # already playing earn more), capped at the class saturation.
        if dt > 0.0 and growth > 0.0:
            earners *= math.exp(growth * dt)
            earners = min(earners, cap_pct / 100.0 * n)
        return earners

    # The next refresh is due one interval after the last one; the
    # checkpoint sits `since_last_hours` into that interval already.
    next_refresh = interval_hours - since_last_hours
    if next_refresh < 0.0:
        next_refresh = 0.0
    t = 0.0
    refresh_no = 0
    while next_refresh <= horizon:
        earners = progress(earners, n, next_refresh - t)
        # --- the refresh itself: a dilution step. ---
        refresh_no += 1
        ratio = 1.0 + (first_ratio - 1.0) * (
            PRIOR_JUMP_RATIO_DECAY ** (refresh_no - 1))
        ratio = max(ratio, PRIOR_JUMP_RATIO_FLOOR)
        n_new = min(n * ratio, N_CAP)
        # The new cohort's rate rises as the week goes on (later
        # joiners have owned the game longer), steeply for early-story
        # trophies -- a day-7 joiner has had days to play, where a
        # launch-night joiner had hours. The rise is capped at 30x:
        # the influx always includes brand-new buyers, so the
        # new-player rate can never reach the whole-population rate.
        rise = min((1.0 + alpha) ** (refresh_no - 1), 30.0)
        new_rate = min(cap_pct, new_base_pct * rise)
        new_rate *= new_mult
        earners += clamp(new_rate, 0.0, 100.0) / 100.0 * (n_new - n)
        n = n_new
        # The refresh itself cannot push the rate past the class
        # saturation either: the newcomers hold at most the cap.
        earners = min(earners, cap_pct / 100.0 * n)
        t = next_refresh
        next_refresh += interval_hours
    # Progression over the tail after the last refresh.
    earners = progress(earners, n, horizon - t)
    return clamp(earners / n * 100.0, 0.01, 99.99)


def forecast_case(case):
    """Forecast one case; returns the output dict for that case."""
    trophies = case.get("trophies") or []
    n = len(trophies)
    horizon_minutes = float(case.get("horizon_minutes") or 0)
    horizon_hours = horizon_minutes / 60.0
    at = parse_time(case.get("at"))
    rows = read_history(case)
    rates_now = last_rates(rows, n)

    # --- refresh statistics measured from the history itself. ---
    jump_times, ratios, jump_new_rates, last_jump_time = measure_jumps(rows, n)
    if len(jump_times) >= 2:
        gaps = [(b - a).total_seconds() / 3600.0
                for a, b in zip(jump_times, jump_times[1:])]
        interval = median(gaps) or PRIOR_REFRESH_INTERVAL_HOURS
    else:
        interval = PRIOR_REFRESH_INTERVAL_HOURS
    interval = clamp(interval, 1.0, 48.0)
    if ratios:
        first_ratio = median(ratios)
    else:
        first_ratio = PRIOR_FIRST_JUMP_RATIO
    first_ratio = clamp(first_ratio, 1.05, 20.0)

    # Time since the last refresh: from the last jump, else from the
    # first read (the whole flat stretch counts as one interval in
    # progress -- launch night was flat for 3.7 h before its refresh).
    if last_jump_time is not None and at is not None:
        since_last = (at - last_jump_time).total_seconds() / 3600.0
    elif rows and at is not None:
        since_last = (at - rows[0][0]).total_seconds() / 3600.0
    else:
        since_last = 0.0
    since_last = clamp(since_last, 0.0, interval)

    last_players = None
    for _when, players, _rates in reversed(rows):
        if players is not None:
            last_players = players
            break
    if last_players is None:
        last_players = 1000

    growth_obs = observed_growth(rows, n)

    out_rows = []
    for i, trophy in enumerate(trophies):
        klass = classify(trophy, rates_now[i])
        profile = PROFILES[klass]
        growth_h, cap, new_mult, _alpha = profile
        rate = rates_now[i]
        if rate is None:
            rate = 1.0  # no read ever carried this trophy: start neutral
        # New-player rate: measured from the last jump when the history
        # has one, blended toward the class prior; else the prior alone.
        prior_new = new_mult * rate
        if jump_new_rates is not None and jump_new_rates[i] is not None:
            new_base = 0.65 * jump_new_rates[i] + 0.35 * prior_new
        else:
            new_base = prior_new
        new_base = clamp(new_base, 0.0, cap * 100.0)
        # Earner growth: the class prior, blended (geometric mean) with
        # a measured rise on flat stretches when one exists, kept
        # within 0.25x-4x of the prior so one noisy stretch cannot
        # dominate.
        growth = growth_h
        if growth_obs[i] is not None and growth_obs[i] > 0.0:
            blended = math.sqrt(growth_h * growth_obs[i])
            growth = clamp(blended, 0.25 * growth_h, 4.0 * growth_h)

        scenario_rates = []
        for shift_frac, new_factor, growth_factor in SCENARIOS:
            # The opening trophy is different: nearly every new player
            # holds it (~99%), so its new-player rate barely varies
            # between scenarios -- applying the generic 0.6x/1.7x swing
            # to it would invent uncertainty that does not exist.
            if klass == "first":
                new_factor = clamp(new_factor, 0.98, 1.01)
                growth_factor = 1.0
                shift_frac = 0.0
            scenario_rates.append(simulate(
                rate, last_players, profile, new_base, growth,
                horizon_hours, since_last, interval, first_ratio,
                shift_frac * interval, new_factor, growth_factor))
        scenario_rates.sort()

        # Widen the outer quantiles a little for the part of the future
        # no refresh count can capture, growing with the horizon.
        extra = SPREAD_PER_SQRT_DAY * math.sqrt(max(horizon_hours, 0.0) / 24.0)
        quantiles = []
        for tau, z, centre in zip(TAUS, Z_SCORES, scenario_rates):
            q = inv_logit(logit(centre / 100.0) + z * extra) * 100.0
            quantiles.append(q)
        # Keep the quantiles ordered and inside the legal range.
        cleaned = []
        prev = 0.01
        for q in quantiles:
            q = clamp(round(q, 4), 0.01, 99.99)
            if q < prev:
                q = prev
            cleaned.append(q)
            prev = q
        out_rows.append({"id": trophy.get("id"), "q": cleaned})
    return {"trophies": out_rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(in_dir, "*.in"))):
        with open(path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_path = os.path.join(out_dir, "%s.out" % case.get("id"))
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(result, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
