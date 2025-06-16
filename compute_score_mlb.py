import pandas as pd
import csv
import random
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, LpStatus, HiGHS_CMD
import os
import numpy as np

# --- USER: Enter the date here ---
date = "6_13"  # Change this to update the date in filenames
# ----------------------------------------

def load_data(hitters_csv, pitchers_csv):
    # Load and clean hitters data
    hitters = pd.read_csv(hitters_csv)
    hitters = hitters[["Player Name", "Pos", "Team", "Salary", "Proj FP", "Actual FP"]]
    
    # Load and clean pitchers data
    pitchers = pd.read_csv(pitchers_csv)
    pitchers = pitchers[["Player Name", "Team", "Salary", "Proj FP", "Actual FP"]]
    pitchers["Pos"] = "P"
    
    # Combine and clean
    df = pd.concat([hitters, pitchers], ignore_index=True)
    return df.dropna(subset=["Player Name", "Salary", "Pos", "Proj FP", "Team"])

def get_unused_player_combinations(df, previous_lineups, num_forced=1):
    """Get players that haven't been used together in previous lineups"""
    if not previous_lineups:
        return random.sample(list(df["Player Name"]), num_forced)
    
    # Track player combinations that have been used
    used_combinations = set()
    for lineup in previous_lineups:
        if num_forced == 1:
            used_combinations.update(lineup)
        else:
            # Add all combinations of size num_forced from this lineup
            lineup_list = list(lineup)
            for i in range(len(lineup_list)):
                for j in range(i + 1, len(lineup_list)):
                    if num_forced == 2:
                        used_combinations.add(tuple(sorted([lineup_list[i], lineup_list[j]])))
                    else:
                        for k in range(j + 1, len(lineup_list)):
                            used_combinations.add(tuple(sorted([lineup_list[i], lineup_list[j], lineup_list[k]])))
    
    # Try to find unused combination
    max_attempts = 100
    for _ in range(max_attempts):
        selected = tuple(sorted(random.sample(list(df["Player Name"]), num_forced)))
        if selected not in used_combinations:
            return list(selected)
    
    return None  # No unused combinations found

def optimize_lineup(df, previous_lineups, forced_players=None, salary_cap=50000, min_variance_multiplier=2):
    players = list(df["Player Name"])
    x = LpVariable.dicts("select", players, cat=LpBinary)
    model = LpProblem("mlb_Lineup_Optimization", LpMaximize)

    # Objective: Maximize projected fantasy points
    proj_fp = dict(zip(df["Player Name"], df["Proj FP"]))
    model += lpSum([proj_fp[p] * x[p] for p in players])

    # Force selected players if any
    if forced_players:
        for player in forced_players:
            model += x[player] == 1

    # Basic constraints
    salary = dict(zip(df["Player Name"], df["Salary"]))
    model += lpSum([salary[p] * x[p] for p in players]) <= salary_cap
    
    # Position constraints
    pitcher = df[df["Pos"].str.contains("P")]["Player Name"]
    catcher = df[df["Pos"].str.contains("C")]["Player Name"]
    firstbase = df[df["Pos"].str.contains("1B")]["Player Name"]
    secondbase = df[df["Pos"].str.contains("2B")]["Player Name"]
    thirdbase = df[df["Pos"].str.contains("3B")]["Player Name"]
    shortstop = df[df["Pos"].str.contains("SS")]["Player Name"]
    outfield = df[df["Pos"].str.contains("OF")]["Player Name"]
    
    model += lpSum([x[p] for p in pitcher]) == 2, "Pitcher"
    model += lpSum([x[p] for p in catcher]) == 1, "Catcher"
    model += lpSum([x[p] for p in firstbase]) == 1, "First Base"
    model += lpSum([x[p] for p in secondbase]) == 1, "Second Base"
    model += lpSum([x[p] for p in thirdbase]) == 1, "Third Base"
    model += lpSum([x[p] for p in shortstop]) == 1, "Shortstop"
    model += lpSum([x[p] for p in outfield]) == 3, "Outfield"
    model += lpSum([x[p] for p in players]) == 10, "TotalPlayers"

    # Team stacking
    team_players = {team: list(group["Player Name"]) 
                   for team, group in df[df["Pos"].str.contains("C|1B|2B|3B|SS|OF")].groupby("Team")}
    
    team_flags = LpVariable.dicts("team_used", team_players.keys(), cat=LpBinary)
    for team, plist in team_players.items():
        for p in plist:
            model += x[p] <= team_flags[team]
    model += lpSum(team_flags.values()) <= 4

    # Enforce uniqueness from previous lineups
    for i, prev_lineup in enumerate(previous_lineups):
        model += lpSum([x[p] for p in prev_lineup]) <= 7, f"Unique_{i}"

    # Consecutive hitters constraint
    for team in df['Team'].unique():
        team_hitters = df[(df['Team'] == team) & (~df['Pos'].str.contains('P'))]
        if len(team_hitters) >= 3:
            # Create a binary variable for each team to track if we use their hitters
            team_hitter_flag = LpVariable(f"team_{team}_hitters", cat=LpBinary)
            # If we use this team's hitters, we must use at least 2 of their first 3
            hitters = team_hitters.head(3)['Player Name'].tolist()
            if len(hitters) == 3:
                model += x[hitters[0]] + x[hitters[1]] + x[hitters[2]] >= 2 * team_hitter_flag
                # We must use at least one team's hitters
                if team == list(df['Team'].unique())[0]:  # Only add for first team
                    model += team_hitter_flag == 1

    # Variance constraint: total variance must be at least min_variance_multiplier times the median variance
    median_variance = variance_df["Variance"].median()
    min_required_variance = min_variance_multiplier * median_variance
    model += lpSum([variance_dict.get(p, median_variance) * x[p] for p in players]) >= min_required_variance, "MinVariance"

    # Solve
    model.solve(HiGHS_CMD(msg=0))
    
    if LpStatus[model.status] != "Optimal":
        return None

    selected = [p for p in players if x[p].varValue == 1]
    
    # Verify lineup composition
    if len(selected) != 10:
        return None
        
    # Verify position requirements
    selected_df = df[df["Player Name"].isin(selected)]
    num_pitchers = len(selected_df[selected_df["Pos"] == "P"])
    num_catchers = len(selected_df[selected_df["Pos"] == "C"])
    num_firstbase = len(selected_df[selected_df["Pos"] == "1B"])
    num_secondbase = len(selected_df[selected_df["Pos"] == "2B"])
    num_thirdbase = len(selected_df[selected_df["Pos"] == "3B"])
    num_shortstop = len(selected_df[selected_df["Pos"] == "SS"])
    num_outfield = len(selected_df[selected_df["Pos"] == "OF"])
    
    if (num_pitchers != 2 or num_catchers != 1 or num_firstbase != 1 or 
        num_secondbase != 1 or num_thirdbase != 1 or num_shortstop != 1 or 
        num_outfield != 3):
        return None
    
    # Double-check uniqueness
    selected_set = frozenset(selected)
    if any(frozenset(prev) == selected_set for prev in previous_lineups):
        return None

    # Verify variance constraint is met
    total_variance = sum(variance_dict.get(player, median_variance) for player in selected)
    if total_variance < min_required_variance:
        print(f"❌ Variance constraint not met: {total_variance:.2f} < {min_required_variance:.2f}")
        return None
        
    return selected

# Create output folders
os.makedirs("mlb_lineups", exist_ok=True)
os.makedirs("mlb_summaries", exist_ok=True)

# Main execution
hitters_file = f"mlb_csvs/DFN MLB Hitters DK {date}.csv"
pitchers_file = f"mlb_csvs/DFN MLB Pitchers DK {date}.csv"
df = load_data(hitters_file, pitchers_file)

# Load player variance data
variance_df = pd.read_csv("player_variance.csv")
# Replace NaN and inf values with NaN
variance_df['Variance'] = variance_df['Variance'].replace([np.inf, -np.inf], np.nan)
# Calculate median from non-zero variance values
non_zero_variance = variance_df[variance_df['Variance'] > 0]['Variance']
median_variance = non_zero_variance.median()
# Fill NaN values with median variance
variance_df['Variance'] = variance_df['Variance'].fillna(median_variance)
variance_dict = dict(zip(variance_df["Player Name"], variance_df["Variance"]))

# Set variance to median variance for players not found in variance_df
for player in df["Player Name"]:
    if player not in variance_dict:
        variance_dict[player] = median_variance

# Print variance statistics
print("\nVariance Statistics:")
print(f"Number of players with non-zero variance: {len(non_zero_variance)}")
print(f"Median variance (from non-zero values): {median_variance:.2f}")
print(f"Minimum required variance: {2 * median_variance:.2f}")

lineups = []
desired_lineups = 150
lineup_output_file = f"mlb_lineups/draftkings_mlb_lineups_{date}.csv"
summary_output_file = f"mlb_summaries/draftkings_mlb_lineup_summaries_{date}.csv"

with open(lineup_output_file, mode="w", newline='') as lineup_file, \
     open(summary_output_file, mode="w", newline='') as summary_file:
    
    lineup_writer = csv.writer(lineup_file)
    summary_writer = csv.writer(summary_file)
    
    # Headers
    lineup_writer.writerow(["Lineup #", "Player Name", "Position", "Team", "Salary", "Proj FP", "Actual FP"])
    summary_writer.writerow(["Lineup #", "Total Salary", "Total Proj FP", "Total Actual FP", "Total Variance"])
    
    forced_count = 1
    attempts_without_success = 0
    max_attempts_without_success = 50

    while len(lineups) < desired_lineups and attempts_without_success < max_attempts_without_success:
        # Get players to force into lineup
        forced_players = get_unused_player_combinations(df, lineups, forced_count)
        if not forced_players:
            if forced_count < 3:
                print(f"\nNo more unique combinations with {forced_count} players, trying {forced_count + 1} players")
                forced_count += 1
                attempts_without_success = 0
                continue
            else:
                print(f"\nNo more unique combinations possible with {forced_count} players")
                break
                
        print(f"\nTrying lineup {len(lineups) + 1} - Forcing: {', '.join(forced_players)}")
        result = optimize_lineup(df, lineups, forced_players)
        
        if result:
            # Verify uniqueness one more time
            if any(set(result) == set(prev) for prev in lineups):
                print("❌ Duplicate lineup detected, skipping")
                attempts_without_success += 1
                continue
                
            lineups.append(result)
            lineup_df = df[df["Player Name"].isin(result)].copy()
            total_salary = lineup_df["Salary"].sum()
            total_proj_fp = lineup_df["Proj FP"].sum()
            total_actual_fp = lineup_df["Actual FP"].sum()
            total_variance = sum(variance_dict.get(player, median_variance) for player in result)
            
            # Write full player details
            for _, row in lineup_df.iterrows():
                lineup_writer.writerow([
                    len(lineups),
                    row["Player Name"],
                    row["Pos"],
                    row["Team"],
                    row["Salary"],
                    row["Proj FP"],
                    row["Actual FP"]
                ])
            
            # Write summary
            summary_writer.writerow([len(lineups), total_salary, total_proj_fp, total_actual_fp, total_variance])
            print(f"✅ Lineup {len(lineups)} generated (Salary: ${total_salary:,}, Proj FP: {total_proj_fp:.2f}, Actual FP: {total_actual_fp:.2f}, Variance: {total_variance:.2f})")
            attempts_without_success = 0
        else:
            print(f"❌ Could not generate valid lineup with forced players")
            attempts_without_success += 1
            
        if attempts_without_success >= max_attempts_without_success:
            if forced_count < 3:
                print(f"\nToo many failed attempts, increasing forced players from {forced_count} to {forced_count + 1}")
                forced_count += 1
                attempts_without_success = 0
            else:
                print("\nToo many failed attempts with maximum forced players, stopping")
                break

print(f"\n📊 Generated {len(lineups)} unique lineups")
