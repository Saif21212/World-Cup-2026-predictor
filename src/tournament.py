"""
Full 2026 FIFA World Cup tournament simulator.

Pipeline:
  1. Group stage  — all 12 groups play a round-robin (simulate_group).
  2. Qualification — top 2 from each group (24) + 8 best third-place teams = 32.
  3. Knockout      — single-elimination: R32 → R16 → QF → SF → Final.

MATCH ENGINE (Session 6 upgrade):
  All matches now use simulate_score() from poisson.py which samples real
  scorelines (goals_a, goals_b) from Poisson distributions calibrated on
  historical data with an opponent-adjusted iterative solver.
  Group tables use REAL goal difference and goals-for.
  Knockout draws (equal scores) are resolved by a weighted Elo coin-flip
  (standing in for extra time and penalties).

KNOWN REMAINING SIMPLIFICATIONS:
  - Bracket seeding: simple positional scheme, not the official 2026 matrix.
  - All host group/knockout matches assumed to be on home soil.
"""

import random
from src.predict        import build_model, match_probabilities
from src.poisson        import simulate_score, compute_team_strengths
from src.group_stage    import simulate_group
from src.teams_2026     import GROUPS
from src.home_advantage import compute_host_bonuses


# ---------------------------------------------------------------------------
# Step 1 — Group stage → 32 qualified teams
# ---------------------------------------------------------------------------

def simulate_group_stage(
    ratings:      dict[str, float],
    attack:       dict[str, float],
    defense:      dict[str, float],
    mu:           float,
    host_bonuses: dict[str, float] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """
    Run all 12 groups and return the 32 qualifying teams.

    Returns
    -------
    winners     : 12 group winners    (group order A→L)
    runners_up  : 12 group runners-up (group order A→L)
    best_thirds : 8 best third-place teams, ranked by points then GD
    """
    winners:       list[str]  = []
    runners_up:    list[str]  = []
    third_records: list[dict] = []

    for label, teams in GROUPS.items():
        standings = simulate_group(
            teams, ratings, attack, defense, mu, host_bonuses=host_bonuses
        )
        winners.append(standings[0]["team"])
        runners_up.append(standings[1]["team"])
        third_records.append(standings[2])

    # Best-8-thirds rule: rank by points, then goal difference, then goals for
    third_records_sorted = sorted(
        third_records,
        key=lambda r: (r["points"], r["gd"], r["gf"]),
        reverse=True,
    )
    best_thirds = [r["team"] for r in third_records_sorted[:8]]

    qualified = winners + runners_up + best_thirds
    assert len(qualified) == 32, f"Expected 32 qualified teams, got {len(qualified)}"
    return winners, runners_up, best_thirds


# ---------------------------------------------------------------------------
# Step 2a — Knockout: one match → one winner
# ---------------------------------------------------------------------------

def knockout_match(
    team_a: str,
    team_b: str,
    ratings:      dict[str, float],
    attack:       dict[str, float],
    defense:      dict[str, float],
    mu:           float,
    host_bonuses: dict[str, float] | None = None,
) -> str:
    """
    Play a single knockout match; return the winner's name.

    Simulates a real scoreline via Poisson.  If the score is level after
    90 minutes (g_a == g_b), resolves with a weighted Elo coin-flip
    (extra-time / penalties placeholder — see session notes).
    """
    _bonuses = host_bonuses or {}
    g_a, g_b = simulate_score(
        team_a, team_b, attack, defense, mu, host_bonuses=_bonuses
    )

    if g_a > g_b:
        return team_a
    if g_b > g_a:
        return team_b

    # --- Draw: weighted Elo coin-flip ---
    bonus_a = _bonuses.get(team_a, 0.0)
    bonus_b = _bonuses.get(team_b, 0.0)
    r_a = ratings[team_a]
    r_b = ratings[team_b]
    p_win_a, _, p_win_b = match_probabilities(
        r_a, r_b,
        home_bonus_a=bonus_a,
        home_bonus_b=bonus_b,
    )
    return random.choices([team_a, team_b], weights=[p_win_a, p_win_b], k=1)[0]


# ---------------------------------------------------------------------------
# Step 2b — Seeding + bracket play
# ---------------------------------------------------------------------------

def _seed_bracket(
    winners:     list[str],
    runners_up:  list[str],
    best_thirds: list[str],
) -> list[tuple[str, str]]:
    """
    Pair 32 teams into 16 Round-of-32 match-ups.

    Simplified seeding (not the official 2026 slot-assignment matrix):
      Slots  1– 8 : winners[0–7]    vs best_thirds[0–7]
      Slots  9–12 : winners[8–11]   vs runners_up[0–3]
      Slots 13–16 : runners_up[4–7] vs runners_up[8–11]
    """
    matchups: list[tuple[str, str]] = []
    for i in range(8):
        matchups.append((winners[i], best_thirds[i]))
    for i in range(4):
        matchups.append((winners[8 + i], runners_up[i]))
    for i in range(4):
        matchups.append((runners_up[4 + i], runners_up[8 + i]))
    assert len(matchups) == 16
    return matchups


def _play_round(
    teams:        list[str],
    ratings:      dict[str, float],
    attack:       dict[str, float],
    defense:      dict[str, float],
    mu:           float,
    host_bonuses: dict[str, float] | None = None,
) -> list[str]:
    """Play one knockout round; return the winners (half the field)."""
    assert len(teams) % 2 == 0
    round_winners = []
    for i in range(0, len(teams), 2):
        winner = knockout_match(
            teams[i], teams[i + 1],
            ratings, attack, defense, mu, host_bonuses,
        )
        round_winners.append(winner)
    return round_winners


def simulate_knockout(
    winners:      list[str],
    runners_up:   list[str],
    best_thirds:  list[str],
    ratings:      dict[str, float],
    attack:       dict[str, float],
    defense:      dict[str, float],
    mu:           float,
    host_bonuses: dict[str, float] | None = None,
) -> str:
    """Run the full single-elimination bracket and return the champion."""
    matchups = _seed_bracket(winners, runners_up, best_thirds)
    field = [team for pair in matchups for team in pair]   # 32 teams

    for _round in ["Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "Final"]:
        field = _play_round(field, ratings, attack, defense, mu, host_bonuses)

    assert len(field) == 1
    return field[0]


# ---------------------------------------------------------------------------
# Step 3 — Full tournament
# ---------------------------------------------------------------------------

def simulate_tournament(
    ratings:      dict[str, float],
    attack:       dict[str, float],
    defense:      dict[str, float],
    mu:           float,
    host_bonuses: dict[str, float] | None = None,
) -> str:
    """Simulate one complete World Cup; return the champion's name."""
    winners, runners_up, best_thirds = simulate_group_stage(
        ratings, attack, defense, mu, host_bonuses
    )
    return simulate_knockout(
        winners, runners_up, best_thirds,
        ratings, attack, defense, mu, host_bonuses,
    )


# ---------------------------------------------------------------------------
# Run as script — 5 000-sim Monte Carlo with Poisson match engine
# ---------------------------------------------------------------------------

def _run_mc(ratings, attack, defense, mu, host_bonuses, n):
    counts: dict[str, int] = {}
    for _ in range(n):
        champ = simulate_tournament(ratings, attack, defense, mu, host_bonuses)
        counts[champ] = counts.get(champ, 0) + 1
    return counts


def _print_table(counts, n, top=15):
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    print(f"  {'Team':<26}  {'Titles':>6}  {'%':>6}")
    print(f"  {'─'*44}")
    for team, cnt in ranked[:top]:
        print(f"  {team:<26}  {cnt:>6}  {cnt/n*100:>5.1f}%")


if __name__ == "__main__":
    N = 5_000

    print("Building Elo ratings …")
    ratings = build_model()

    print("Computing Poisson strengths …")
    attack, defense, mu = compute_team_strengths(ratings=ratings)

    print("Computing host home-advantage bonuses …")
    host_bonuses, global_bonus = compute_host_bonuses()
    print(f"  Global bonus: {global_bonus:.1f} pts")
    for host, bonus in host_bonuses.items():
        print(f"  {host:<20}: +{bonus:.1f} pts")

    # --- Baseline: no home advantage ---
    print(f"\nRunning {N:,} sims WITHOUT home advantage …")
    counts_base = _run_mc(ratings, attack, defense, mu, host_bonuses=None, n=N)

    # --- With home advantage ---
    print(f"Running {N:,} sims WITH home advantage …")
    counts_home = _run_mc(ratings, attack, defense, mu, host_bonuses=host_bonuses, n=N)

    # --- Top 15 ---
    print(f"\n{'─'*50}")
    print(f"  Top 15 — WITH home advantage, Poisson engine ({N:,} sims)")
    print(f"{'─'*50}")
    _print_table(counts_home, N, top=15)

    # --- Before / After for hosts + Spain ---
    spotlight = ["United States", "Mexico", "Canada", "Spain"]
    print(f"\n{'─'*62}")
    print(f"  Before / After home advantage")
    print(f"  {'Team':<22} {'Without':>8} {'With':>8} {'Change':>8}")
    print(f"  {'─'*50}")
    for team in spotlight:
        b = counts_base.get(team, 0) / N * 100
        h = counts_home.get(team, 0) / N * 100
        sign = "+" if (h - b) >= 0 else ""
        print(f"  {team:<22} {b:>7.1f}%  {h:>7.1f}%  {sign}{h-b:>6.1f}%")
    print(f"{'─'*62}")

    jordan = counts_home.get("Jordan", 0)
    print(f"\n  Jordan: {jordan} titles in {N:,} sims ({jordan/N*100:.2f}%)")
