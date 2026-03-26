import csv
from collections import defaultdict

# 1. Configuration
csv_file = "Arena_Battle_Logs.csv"

# Dictionary to hold the record for each model
# stats[model_name] = {"wins": 0, "losses": 0, "ties": 0, "total": 0}
stats = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0, "total": 0})

print(f"Calculating overall win rates from {csv_file}...\n")

# 2. Parse the CSV
try:
    with open(csv_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_a = row.get("Model_A")
            model_b = row.get("Model_B")
            winner = row.get("Winner")
            
            # Skip empty rows or malformed data
            if not model_a or not model_b or not winner:
                continue
                
            # Increment total matches played for both models
            stats[model_a]["total"] += 1
            stats[model_b]["total"] += 1
            
            # Record the outcome
            if winner == model_a:
                stats[model_a]["wins"] += 1
                stats[model_b]["losses"] += 1
            elif winner == model_b:
                stats[model_b]["wins"] += 1
                stats[model_a]["losses"] += 1
            elif winner.lower() == "tie":
                stats[model_a]["ties"] += 1
                stats[model_b]["ties"] += 1

except FileNotFoundError:
    print(f"ERROR: Could not find '{csv_file}'. Please check the file name.")
    exit()

# 3. Calculate and Sort Results
# Calculate standard Win %: (Wins / Total Matches) * 100
# Note: Ties count toward total matches, pulling down the raw win percentage.
def get_win_percentage(model):
    total = stats[model]["total"]
    if total == 0:
        return 0.0
    return (stats[model]["wins"] / total) * 100

# Sort models from highest win percentage to lowest
sorted_models = sorted(stats.keys(), key=get_win_percentage, reverse=True)

if not sorted_models:
    print("No battle data found in the CSV.")
    exit()

# 4. Print the Leaderboard Dashboard
print(f"{'Model Name':<18} | {'Win %':<8} | {'Wins':<6} | {'Losses':<6} | {'Ties':<6} | {'Total Played'}")
print("-" * 75)

for model in sorted_models:
    s = stats[model]
    win_pct = get_win_percentage(model)
    
    # Strip the "Model_" prefix to keep the output clean
    display_name = model.replace("Model_", "")
    
    print(f"{display_name:<18} | {win_pct:>5.1f}%   | {s['wins']:<6} | {s['losses']:<6} | {s['ties']:<6} | {s['total']}")
print("-" * 75)