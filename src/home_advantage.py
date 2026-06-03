"""
Home-advantage measurement and shrinkage for the 2026 World Cup hosts.

ONLY USA, Mexico, and Canada receive a home boost — they are the three host
nations.  The boost is expressed as extra Elo points added to their effective
rating for any match they play in the tournament.

HOW THE MEASUREMENT WORKS
--------------------------
Plain Elo is fit on ALL matches without a home-advantage term, so its expected
scores in non-neutral matches systematically underestimate the home team.
The gap between actual and expected performance is the "home advantage delta":

    delta = mean(actual_home_score - elo_expected_home_score)
             over all non-neutral matches

Where actual_home_score = 1 (win), 0.5 (draw), or 0 (loss).

Converting delta to Elo points:
  For two equally rated teams (rating_A = rating_B), the expected score is 0.5.
  If the true home-advantage expected score is 0.5 + delta, then the Elo bonus
  b that produces this is found by solving:
      1 / (1 + 10^(−b/400)) = 0.5 + delta
  → b = 400 * log10((0.5 + delta) / (0.5 - delta))

(For unequal teams the mapping is approximate, but this is the standard
 approach used in applied Elo systems including FIFA's own model.)

SHRINKAGE (PARTIAL POOLING)
----------------------------
Each host's own home-advantage estimate is noisy if they have few home matches.
We blend it toward the global baseline using a weighted average:

    shrunk = (n / (n + K)) * own_estimate
           + (K / (n + K)) * global_estimate

  n = number of home matches the host has played
  K = 50  ← shrinkage constant

Interpretation of K:
  K acts as a "virtual sample size" for the global prior.  With n >> K the
  host's own estimate dominates; with n << K we trust the global baseline.
  K=50 means "we need at least ~50 home matches before heavily trusting a
  team's individual number".

Measured from 36 262 non-neutral matches (martj42 dataset, up to 2026-06-27):

  Global home bonus : 80.4 Elo points  (delta = 0.1137)

  Host        |  n   | own bonus | shrunk (K=50)
  ------------|------|-----------|---------------
  United States| 467 |  78.9 pts |  79.1 pts
  Mexico       | 292 | 123.3 pts | 117.3 pts
  Canada       | 143 |  89.9 pts |  87.7 pts

  → Mexico stays well above global (large sample, genuinely strong home record).
  → USA barely moves (own ≈ global already).
  → Canada is pulled partway toward global (modest sample).
"""

import math
import pandas as pd
from src.elo import expected_score, update_ratings, DEFAULT_RATING


# 2026 host nations
HOSTS: list[str] = ["United States", "Mexico", "Canada"]

# Shrinkage constant (see module docstring)
K_SHRINKAGE: int = 50


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _delta_to_elo_bonus(delta: float) -> float:
    """
    Convert a performance-edge delta into Elo points.
    See module docstring for derivation.
    """
    p = 0.5 + delta
    p = max(0.001, min(0.999, p))    # clamp to avoid log(0)
    return 400.0 * math.log10(p / (1.0 - p))


def _shrink(own: float, n: int, global_b: float, K: int = K_SHRINKAGE) -> float:
    """Blend own estimate toward global baseline weighted by sample size."""
    weight = n / (n + K)
    return weight * own + (1.0 - weight) * global_b


# ---------------------------------------------------------------------------
# Main measurement function
# ---------------------------------------------------------------------------

def compute_host_bonuses(
    results_path: str = "data/raw/results.csv",
) -> tuple[dict[str, float], float]:
    """
    Replay the full match history, measure home advantage, apply shrinkage.

    Returns
    -------
    host_bonuses : dict {team_name: elo_bonus}  — one entry per 2026 host
    global_bonus : float                         — global home Elo bonus in points
    """
    df = pd.read_csv(results_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["home_score", "away_score"])

    ratings: dict[str, float] = {}
    rows: list[dict] = []

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        r_h = ratings.get(home, DEFAULT_RATING)
        r_a = ratings.get(away, DEFAULT_RATING)

        # Record the pre-match Elo expected score and the actual result
        exp_h = expected_score(r_h, r_a)
        hs, as_ = int(row["home_score"]), int(row["away_score"])
        actual_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)

        rows.append({
            "home_team": home,
            "neutral"  : bool(row["neutral"]),
            "exp"      : exp_h,
            "actual"   : actual_h,
        })

        # Update ratings (same vanilla Elo — no home bonus baked in)
        r_h, r_a = update_ratings(r_h, r_a, hs, as_)
        ratings[home] = r_h
        ratings[away] = r_a

    data = pd.DataFrame(rows)

    # --- global home advantage: non-neutral matches only ---
    non_neutral = data[~data["neutral"]]
    delta_global = (non_neutral["actual"] - non_neutral["exp"]).mean()
    global_bonus = _delta_to_elo_bonus(delta_global)

    # --- per-host ---
    host_bonuses: dict[str, float] = {}
    for host in HOSTS:
        host_matches = non_neutral[non_neutral["home_team"] == host]
        n = len(host_matches)

        if n > 0:
            delta_host = (host_matches["actual"] - host_matches["exp"]).mean()
            own_bonus  = _delta_to_elo_bonus(delta_host)
        else:
            # No data: fall back entirely to global
            own_bonus = global_bonus

        shrunk = _shrink(own_bonus, n, global_bonus)
        host_bonuses[host] = shrunk

    return host_bonuses, global_bonus


# ---------------------------------------------------------------------------
# Run as script — print the measurement table
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    print("Replaying match history to measure home advantage …\n")

    # Rerun the measurement with verbose output
    df = pd.read_csv("data/raw/results.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["home_score", "away_score"])

    ratings: dict[str, float] = {}
    rows: list[dict] = []

    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        r_h = ratings.get(home, DEFAULT_RATING)
        r_a = ratings.get(away, DEFAULT_RATING)
        exp_h = expected_score(r_h, r_a)
        hs, as_ = int(row["home_score"]), int(row["away_score"])
        actual_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        rows.append({"home_team": home, "neutral": bool(row["neutral"]),
                     "exp": exp_h, "actual": actual_h})
        r_h, r_a = update_ratings(r_h, r_a, hs, as_)
        ratings[home] = r_h; ratings[away] = r_a

    data = pd.DataFrame(rows)
    non_neutral = data[~data["neutral"]]

    delta_global = (non_neutral["actual"] - non_neutral["exp"]).mean()
    global_bonus = _delta_to_elo_bonus(delta_global)

    print(f"Step 1 — Global home advantage")
    print(f"  Non-neutral matches : {len(non_neutral):,}")
    print(f"  Delta (actual−Elo)  : {delta_global:.5f}")
    print(f"  Elo-equivalent bonus: {global_bonus:.1f} points\n")

    print(f"Step 2 — Per-host raw estimates")
    host_raw: dict[str, tuple[int, float, float]] = {}
    for host in HOSTS:
        hm = non_neutral[non_neutral["home_team"] == host]
        n  = len(hm)
        d  = (hm["actual"] - hm["exp"]).mean() if n else 0.0
        b  = _delta_to_elo_bonus(d) if n else global_bonus
        host_raw[host] = (n, d, b)
        print(f"  {host:<20}: n={n:>4}, delta={d:+.5f}, own bonus={b:6.1f} pts")

    print(f"\nStep 3 — Shrinkage toward global (K={K_SHRINKAGE})")
    print(f"\n  {'Host':<20} {'Own (pts)':>10} {'n':>5} {'Weight':>8} {'Shrunk (pts)':>13}")
    print(f"  {'─'*60}")
    host_bonuses, _ = compute_host_bonuses()
    for host in HOSTS:
        n, _, own = host_raw[host]
        w = n / (n + K_SHRINKAGE)
        shrunk = host_bonuses[host]
        print(f"  {host:<20} {own:>10.1f} {n:>5} {w:>8.3f} {shrunk:>13.1f}")
    print(f"\n  (Global baseline: {global_bonus:.1f} pts — used when weight is low)")
