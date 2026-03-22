'''
This file compares models using pairwise analysis in which two models go head-to-head. The models have generated a response to the same question 
from 3_generate_data.py and these are fed through to the Google GenerativeAI API. The judge is Gemini 2.5 Pro which is provided a prompt to evaluate
the responses from two models, provides a winner ("A", "B", or "Tie"), and a reason for the result. These results can then be used to output metrics
such as win rate across all models.
'''

import csv
import json
import time
import random
import itertools
from google import genai
from google.genai import types
from google.genai.errors import APIError

#It is best to use a .env file locally or attach to a cloud secret key manager.
API_KEY = ""
client = genai.Client(api_key=API_KEY)

#Gemini 2.5 Pro is used. Tier 1 will get you a daily API request limit of 1000 which is exactly the
#number of API calls made in this script.
JUDGE_MODEL_NAME = 'gemini-2.5-pro'

#Read output from 3_generate_data.py
input_file = "Model_Comparison_Arena.csv"
log_file = "Arena_Battle_Logs.csv"

#define all models used in pairwise analysis.
MODELS = ["Model_Base", "Model_SFT", "Model_DPO", "Model_GRPO", "Model_SimPO"]

#define Judge prompt
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

#Running of pairwise loop. Each pair of models are evaluated equally over 100 generated responses.
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
        
        #Make call to API. Setting temperature to 0 makes to response more factual.
        success = False
        while not success:
            try:
                response = client.models.generate_content(
                    model=JUDGE_MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
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
                
                #Ensure output is correctly formatted as a dictionary
                if isinstance(verdict, list):
                    verdict = verdict[0]
                
                if verdict["winner"] == "A":
                    actual_winner = model_a_name
                elif verdict["winner"] == "B":
                    actual_winner = model_b_name
                else:
                    actual_winner = "Tie"
                
                #write results to output file.
                with open(log_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([q_idx+1, model_a_name, model_b_name, actual_winner, verdict.get("reasoning")])
                
                print(f"Battle {battle_count}/{total_battles} | {model_a_name} vs {model_b_name} -> Winner: {actual_winner}")
                success = True
            
            #log common errors
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