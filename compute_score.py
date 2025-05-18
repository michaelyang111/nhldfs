import pandas as pd
import csv
from collections import defaultdict
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, LpStatus, HiGHS_CMD
import logging
from datetime import datetime
import sys

# Set up logging to both file and console
class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write to file

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Create a timestamp for the log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"lineup_generation_{timestamp}.log"
sys.stdout = TeeLogger(log_file)

print(f"Starting lineup generation - Log file: {log_file}")
print("=" * 50)

def load_data(skaters_csv, goalies_csv):
    skaters = pd.read_csv(skaters_csv)
    print("Loaded skaters data:")
    print(skaters)
    goalies = pd.read_csv(goalies_csv)
    print("Loaded goalies data:")
    print(goalies)

    skaters.columns = skaters.columns.str.strip()
    goalies.columns = goalies.columns.str.strip()

    # Check if columns match (excluding the 'Pos' column which we handle separately)
    skaters_cols = set(skaters.columns) - {'Pos'}
    goalies_cols = set(goalies.columns) - {'Pos'}
    if skaters_cols != goalies_cols:
        print("Warning: Column mismatch between skaters and goalies files!")
        print("Skaters columns:", sorted(list(skaters_cols)))
        print("Goalies columns:", sorted(list(goalies_cols)))
        print("Difference:", sorted(list(skaters_cols.symmetric_difference(goalies_cols))))

    if "Pos" not in goalies.columns:
        goalies["Pos"] = "G"
    
    # Concatenate the dataframes
    combined_df = pd.concat([skaters, goalies], ignore_index=True)
    result_df = combined_df.dropna(subset=["Player Name", "Salary", "Pos", "Proj FP", "Team"])
    
    print(f"Combined data shape: {result_df.shape}")
    
    # Save the combined data to CSV
    result_df.to_csv("combined_players.csv", index=False)
    print("Combined data saved to 'combined_players.csv'")
    
    return result_df

def optimize_lineup(df, previous_lineups, salary_cap=50000, strict=True):
    # Randomly shuffle the dataframe
    df = df.sample(frac=1, random_state=None).reset_index(drop=True)
    
    players = list(df["Player Name"])
    x = LpVariable.dicts("select", players, cat=LpBinary)
    model = LpProblem("NHL_Lineup_Optimization", LpMaximize)

    proj_fp = dict(zip(df["Player Name"], df["Proj FP"]))
    salary = dict(zip(df["Player Name"], df["Salary"]))
    model += lpSum([proj_fp[p] * x[p] for p in players]), "Total_Projected_FP"

    # Salary cap constraint
    model += lpSum([salary[p] * x[p] for p in players]) <= salary_cap

    # Positional filters
    def pos_filter(pos):
        return df[df["Pos"].str.contains(pos)]["Player Name"].tolist()

    centers = pos_filter("C")
    wingers = pos_filter("W")
    defense = pos_filter("D")
    goalies = pos_filter("G")

    model += lpSum([x[p] for p in centers]) == 2
    model += lpSum([x[p] for p in wingers]) == 3
    model += lpSum([x[p] for p in defense]) == 2
    model += lpSum([x[p] for p in goalies]) == 1
    model += lpSum([x[p] for p in players]) == 8

    # Strict requirement: D must be on PP1
    if strict:
        pp1_d = df[(df["Pos"].str.contains("D")) & (df["PP"].str.contains("P1", na=False))]["Player Name"].tolist()
        if not pp1_d:
            print("⚠️ Warning: No PP1 defensemen found in the data")
        else:
            print(f"Found {len(pp1_d)} PP1 defensemen: {', '.join(pp1_d)}")
            model += lpSum([x[p] for p in pp1_d]) >= 2
        input("Press Enter to continue...")

    # Team stacking: limit to max stacking teams (C/W)
    linesmen_df = df[df["Pos"].str.contains("C|W")]
    team_to_players = defaultdict(list)
    for _, row in linesmen_df.iterrows():
        team_to_players[row["Team"]].append(row["Player Name"])

    team_flags = LpVariable.dicts("team_used", team_to_players.keys(), cat=LpBinary)
    for team, plist in team_to_players.items():
        for p in plist:
            model += x[p] <= team_flags[team]

    max_teams = 2 if strict else 3
    model += lpSum(team_flags.values()) <= max_teams

    # Goalie anti-correlation
    goalie_df = df[df["Pos"] == "G"]
    for _, goalie in goalie_df.iterrows():
        opp_team = goalie["Opp"]
        g_name = goalie["Player Name"]
        if pd.isna(opp_team):
            continue
        conflict_players = linesmen_df[linesmen_df["Team"] == opp_team]["Player Name"].tolist()
        if conflict_players:
            model += x[g_name] + lpSum([x[p] for p in conflict_players]) <= 1

    model.solve(HiGHS_CMD(msg=0))
    
    # Debug information
    print("\n🔍 Solver Status:", LpStatus[model.status])
    if LpStatus[model.status] != "Optimal":
        print("❌ No valid solution found. Constraints that might be causing issues:")
        if strict:
            print("- Requiring 2 PP1 defensemen")
        print(f"- Maximum {max_teams} teams for stacking")
        print("- Salary cap:", salary_cap)
        input("Press Enter to continue...")
        return None

    selected = [p for p in players if x[p].varValue == 1]
    selected_df = df[df["Player Name"].isin(selected)].copy()
    
    # Print detailed lineup information
    total_salary = selected_df["Salary"].sum()
    total_fp = selected_df["Proj FP"].sum()
    print("\n📊 Potential Lineup Details:")
    print(f"Total Salary: ${total_salary:,.0f}")
    print(f"Total Projected FP: {total_fp:.2f}")
    input("Press Enter to continue...")
    
    # Check if lineup is unique
    if frozenset(selected) in previous_lineups:
        print("❌ Lineup rejected: Duplicate of previous lineup")
        input("Press Enter to continue...")
        return None
        
    # Print team stacking information
    teams_used = selected_df["Team"].value_counts()
    print("\n🏒 Team Stacking:")
    for team, count in teams_used.items():
        print(f"{team}: {count} players")
    input("Press Enter to continue...")
    
    return frozenset(selected)

# ------------------- MAIN ----------------------

skaters_file = "DFN NHL Skaters DK 3_1.csv"
goalies_file = "DFN NHL Goalies DK 3_1.csv"
df = load_data(skaters_file, goalies_file)

# Set random seed for reproducibility if needed
# import random
# random.seed(42)
# np.random.seed(42)

lineups = set()
desired = 50

details_file = "draftkings_nhl_lineups.csv"
summary_file = "draftkings_nhl_lineup_summaries.csv"

with open(details_file, mode="w", newline='') as f_detail, \
     open(summary_file, mode="w", newline='') as f_summary:

    writer_detail = csv.writer(f_detail)
    writer_summary = csv.writer(f_summary)

    writer_detail.writerow(["Lineup #", "Player Name", "Position", "Team", "Salary", "Proj FP"])
    writer_summary.writerow(["Lineup #", "Total Salary", "Total Proj FP"])

    strict_mode = True
    i = 0
    attempts = 0
    max_attempts = 100  # Prevent infinite loops

    while len(lineups) < desired and attempts < max_attempts:
        print(f"\n📝 Attempt {attempts + 1} (Generated {len(lineups)} lineups so far)")
        print(f"Mode: {'Strict' if strict_mode else 'Relaxed'}")
        input("Press Enter to try next lineup...")
        
        result = optimize_lineup(df.copy(), lineups, strict=strict_mode)  # Pass a copy to prevent modifying original
        attempts += 1
        
        if result is None:
            if attempts >= max_attempts:
                print(f"\n⚠️ Reached maximum attempts ({max_attempts}). Stopping.")
            else:
                print("\n⚠️ Could not generate more unique lineups. Ending early.")
            input("Press Enter to see final results...")
            break

        if result in lineups:
            continue

        lineups.add(result)
        lineup_df = df[df["Player Name"].isin(result)].copy()

        total_salary = lineup_df["Salary"].sum()
        total_fp = lineup_df["Proj FP"].sum()

        for _, row in lineup_df.iterrows():
            writer_detail.writerow([
                i + 1,
                row["Player Name"],
                row["Pos"],
                row["Team"],
                row["Salary"],
                row["Proj FP"]
            ])
        writer_summary.writerow([i + 1, total_salary, total_fp])
        print(f"\n✅ Successfully generated lineup #{i + 1}")
        print(f"💰 Salary: ${total_salary:,.0f}")
        print(f"📈 Projected FP: {total_fp:.2f}")
        input("Press Enter to continue...")
        i += 1

print(f"\n🎉 Final Results:")
print(f"Generated {len(lineups)} unique lineups")
print(f"Total attempts: {attempts}")
print(f"Success rate: {(len(lineups)/attempts*100):.1f}%")
print(f"Details saved to: {details_file}")
print(f"Summary saved to: {summary_file}")
input("Press Enter to exit...")
