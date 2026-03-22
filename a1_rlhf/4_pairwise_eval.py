import os
import csv
import json
import time
import random
import itertools
from google import genai
from google.genai import types
from google.genai.errors import APIError

# 1. Configuration
client = genai.Client(api_key="REDACTED_GOOGLE_API_KEY")

# Using 2.5 Pro (The Tier 1 Powerhouse)
JUDGE_MODEL_NAME = 'gemini-2.5-pro'

input_file = "Model_Comparison_Arena.csv"
log_file = "Arena_Battle_Logs.csv"

MODELS = ["Model_Base", "Model_SFT", "Model_DPO", "Model_GRPO", "Model_SimPO"]

elo_ratings = {model: 1200 for model in MODELS}
K_FACTOR = 32 

# 2. The 1v1 Judge Prompt
JUDGE_PROMPT = """You are an expert Cybersecurity Assessor acting as a blind judge.
Read the Instruction and the Expected Human Answer. Then, evaluate the answers from Model A and Model B.
Which model provides a more factually accurate, safer, and logically sound response? 

[Instruction]
{instruction}

[Expected Human Answer]
{expected}

[Model A]
{model_a_answer}

[Model B]
{model_b_answer}

Output your verdict ONLY as a valid JSON object. 
The "winner" field must be exactly "A", "B", or "Tie".
"""

def update_elo(player_a, player_b, winner):
    r_a = elo_ratings[player_a]
    r_b = elo_ratings[player_b]
    
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    expected_b = 1 / (1 + 10 ** ((r_a - r_b) / 400))
    
    if winner == "A":
        score_a, score_b = 1.0, 0.0
    elif winner == "B":
        score_a, score_b = 0.0, 1.0
    else: 
        score_a, score_b = 0.5, 0.5
        
    elo_ratings[player_a] = round(r_a + K_FACTOR * (score_a - expected_a))
    elo_ratings[player_b] = round(r_b + K_FACTOR * (score_b - expected_b))

# 3. Main Tournament Loop
print("Initializing Chatbot Arena Pairwise Tournament...")

with open(input_file, 'r', encoding='utf-8') as f:
    questions = list(csv.DictReader(f))

with open(log_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Question", "Model_A", "Model_B", "Winner", "Reasoning"])

battle_count = 0
total_battles = len(questions) * len(list(itertools.combinations(MODELS, 2)))

for q_idx, row in enumerate(questions):
    matchups = list(itertools.combinations(MODELS, 2))
    
    for contender_1, contender_2 in matchups:
        battle_count += 1
        
        players = [contender_1, contender_2]
        random.shuffle(players)
        model_a_name, model_b_name = players[0], players[1]
        
        prompt = JUDGE_PROMPT.format(
            instruction=row["prompt"],
            expected=row["expected_answer"],
            model_a_answer=row[model_a_name],
            model_b_answer=row[model_b_name]
        )
        
        # 4. API Call
        success = False
        while not success:
            try:
                response = client.models.generate_content(
                    model=JUDGE_MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                        # ARMOR LAYER 1: Force the API to return a strict dictionary schema
                        response_schema={
                            "type": "OBJECT",
                            "properties": {
                                "winner": {"type": "STRING"},
                                "reasoning": {"type": "STRING"}
                            },
                            "required": ["winner", "reasoning"]
                        }
                    )
                )
                
                verdict = json.loads(response.text)
                
                # ARMOR LAYER 2: The Python fallback catch
                # If the model still somehow sneaks a list past the schema, we grab the first item
                if isinstance(verdict, list):
                    verdict = verdict[0]
                
                # Now it is mathematically guaranteed to be a dictionary
                if verdict["winner"] == "A":
                    actual_winner = model_a_name
                elif verdict["winner"] == "B":
                    actual_winner = model_b_name
                else:
                    actual_winner = "Tie"
                    
                update_elo(model_a_name, model_b_name, verdict["winner"])
                
                with open(log_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([q_idx+1, model_a_name, model_b_name, actual_winner, verdict.get("reasoning")])
                
                print(f"Battle {battle_count}/{total_battles} | {model_a_name} vs {model_b_name} -> Winner: {actual_winner}")
                success = True
                
            except APIError as e:
                if e.code == 429:
                    print("  [Rate Limit Hit] Sleeping for 60 seconds...")
                    time.sleep(60)
                else:
                    print(f"  [API Error] Battle failed: {e}. Skipping...")
                    success = True 
            except Exception as e:
                print(f"  [Error] Battle failed: {e}. Skipping...")
                success = True

# 5. Print the Final Leaderboard
print("\n" + "="*50)
print("TOURNAMENT COMPLETE! FINAL ELO LEADERBOARD:")
print("="*50)
sorted_leaderboard = sorted(elo_ratings.items(), key=lambda item: item[1], reverse=True)
for rank, (model, elo) in enumerate(sorted_leaderboard, 1):
    print(f"{rank}. {model.replace('Model_', '')}: {elo} Elo")
print("="*50)
