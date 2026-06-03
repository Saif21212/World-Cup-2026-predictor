"""
Group stage simulator for the 2026 FIFA World Cup.

Each group has 4 teams who play each other once (round robin = 6 matches).
Points: win = 3, draw = 1, loss = 0.

GOAL DIFFERENCE:
  Session 6 upgrade: simulate_score() from poisson.py now returns actual
  scorelines (goals_a, goals_b) sampled from Poisson distributions.
  The group table uses REAL goal difference and goals-for/against.
  The old +1/−1 proxy has been removed.
"""

import random
from itertools import combinations
from src.predict  import build_model
from src.poisson  import simulate_score, compute_team_strengths


# ---------------------------------------------------------------------------
# Single group simulation
# ---------------------------------------------------------------------------

def simulate_group(
    group_teams: list[str],
    ratings:     dict[str, float],
    attack:      dict[str, float],
    defense:     dict[str, float],
    mu:          float,
    host_bonuses: dict[str, float] | None = None,
) -> list[dict]:
    """
    Play a full round-robin within a group and return final standings.

    Parameters
    ----------
    group_teams  : list of exactly 4 team names
    ratings      : Elo ratings dict (used by simulate_score for draw resolution)
    attack       : Poisson attack ratios from compute_team_strengths()
    defense      : Poisson defense ratios from compute_team_strengths()
    mu           : global average goals per team per game
    host_bonuses : optional {team: elo_bonus} for host teams playing at home

    Returns
    -------
    standings : list of dicts sorted by (points desc, gd desc, gf desc).
                Keys: team, played, won, drawn, lost, gf, ga, gd, points.
    """
    _bonuses = host_bonuses or {}

    records: dict[str, dict] = {
        team: {
            "team": team, "played": 0,
            "won": 0, "drawn": 0, "lost": 0,
            "gf": 0, "ga": 0, "gd": 0, "points": 0,
        }
        for team in group_teams
    }

    # Play every unique pair exactly once (C(4,2) = 6 matches)
    for team_a, team_b in combinations(group_teams, 2):
        g_a, g_b = simulate_score(
            team_a, team_b, attack, defense, mu,
            host_bonuses=_bonuses,
        )

        records[team_a]["played"] += 1
        records[team_b]["played"] += 1
        records[team_a]["gf"] += g_a
        records[team_a]["ga"] += g_b
        records[team_b]["gf"] += g_b
        records[team_b]["ga"] += g_a
        records[team_a]["gd"] += g_a - g_b
        records[team_b]["gd"] += g_b - g_a

        if g_a > g_b:
            records[team_a]["won"]    += 1
            records[team_b]["lost"]   += 1
            records[team_a]["points"] += 3
        elif g_a == g_b:
            records[team_a]["drawn"]   += 1
            records[team_b]["drawn"]   += 1
            records[team_a]["points"]  += 1
            records[team_b]["points"]  += 1
        else:
            records[team_b]["won"]    += 1
            records[team_a]["lost"]   += 1
            records[team_b]["points"] += 3

    standings = sorted(
        records.values(),
        key=lambda r: (r["points"], r["gd"], r["gf"]),
        reverse=True,
    )
    return standings


# ---------------------------------------------------------------------------
# Pretty-print a group table
# ---------------------------------------------------------------------------

def print_standings(standings: list[dict], group_label: str = "") -> None:
    """Print a group table with real GF, GA, GD."""
    header = f"Group {group_label}" if group_label else "Group"
    print(f"\n{'─'*58}")
    print(f"  {header}")
    print(f"  {'Team':<26} {'P':>2} {'W':>2} {'D':>2} {'L':>2} {'GF':>4} {'GA':>4} {'GD':>4} {'Pts':>4}")
    print(f"{'─'*58}")
    for i, r in enumerate(standings, 1):
        qualifier = "Q" if i <= 2 else " "
        print(
            f"  {qualifier}{i}. {r['team']:<24}"
            f" {r['played']:>2} {r['won']:>2} {r['drawn']:>2} {r['lost']:>2}"
            f" {r['gf']:>4} {r['ga']:>4} {r['gd']:>4} {r['points']:>4}"
        )
    print(f"{'─'*58}")


# ---------------------------------------------------------------------------
# Monte Carlo: simulate a group N times, tally finish positions
# ---------------------------------------------------------------------------

def simulate_group_many(
    group_teams: list[str],
    ratings:     dict[str, float],
    attack:      dict[str, float],
    defense:     dict[str, float],
    mu:          float,
    n:           int = 1000,
    host_bonuses: dict[str, float] | None = None,
) -> dict[str, dict[int, int]]:
    """
    Run simulate_group() n times and count finish positions (1st–4th).

    Returns
    -------
    tallies : dict {team: {1: count, 2: count, 3: count, 4: count}}
    """
    tallies: dict[str, dict[int, int]] = {
        team: {1: 0, 2: 0, 3: 0, 4: 0} for team in group_teams
    }
    for _ in range(n):
        standings = simulate_group(
            group_teams, ratings, attack, defense, mu, host_bonuses=host_bonuses
        )
        for pos, record in enumerate(standings, 1):
            tallies[record["team"]][pos] += 1
    return tallies


def print_monte_carlo(
    tallies: dict[str, dict[int, int]],
    n: int,
    group_label: str = "",
) -> None:
    """Print finish-position probabilities from a Monte Carlo run."""
    header = f"Group {group_label} — {n:,} simulations" if group_label else f"{n:,} simulations"
    print(f"\n{'─'*58}")
    print(f"  {header}")
    print(f"  {'Team':<26} {'1st':>6} {'2nd':>6} {'3rd':>6} {'4th':>6}")
    print(f"{'─'*58}")
    sorted_teams = sorted(tallies.keys(), key=lambda t: tallies[t][1], reverse=True)
    for team in sorted_teams:
        counts = tallies[team]
        print(
            f"  {team:<26}"
            f" {counts[1]/n*100:5.1f}%"
            f" {counts[2]/n*100:5.1f}%"
            f" {counts[3]/n*100:5.1f}%"
            f" {counts[4]/n*100:5.1f}%"
        )
    print(f"{'─'*58}")


# ---------------------------------------------------------------------------
# Run as script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.teams_2026     import GROUPS
    from src.home_advantage import compute_host_bonuses

    print("Building Elo ratings and Poisson strengths …")
    ratings = build_model()
    attack, defense, mu = compute_team_strengths(ratings=ratings)
    host_bonuses, _ = compute_host_bonuses()

    N = 1000
    for label, teams in GROUPS.items():
        standings = simulate_group(teams, ratings, attack, defense, mu, host_bonuses)
        print_standings(standings, group_label=label)
        tallies = simulate_group_many(
            teams, ratings, attack, defense, mu, n=N, host_bonuses=host_bonuses
        )
        print_monte_carlo(tallies, n=N, group_label=label)
