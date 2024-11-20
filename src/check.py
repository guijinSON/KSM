import pandas as pd
import os
import argparse
from data import generate_queries_local, generate_queries_litellm, parse_boxed_value, answer_in_last_sentence, parse_mcqa_value, parse_ksm_value
import json

def check_func(checking):
    if checking:
        return "O"
    else:
        return "X"


parser = argparse.ArgumentParser(description="Generate predictions and save results to CSV.")
parser.add_argument('--model_name', type=str, default='gpt-4o',
                    help="Name of the model to use for generating predictions.")
parser.add_argument('--lang_type', type=str, default="ko",
                    help="Language type of the evaluation sets.")
args = parser.parse_args()

model_name = args.model_name
lang = args.lang_type
file_path = os.path.join(f"{lang}_results", model_name.replace("/", "_"))
data_list = ["GSM8K", "MATH", "OMNI_MATH", "MMMLU", "KSM"]
scores = {}

os.makedirs(f"{lang}_check_results/{model_name.replace('/', '_')}", exist_ok=True)

for d in data_list:
    df_result = pd.read_csv(os.path.join(file_path, f"{d}.csv"))
    checks = []
    if d in ["GSM8K", "MATH", "OMNI_MATH"]:
        score = sum([1 for _,row in df_result.iterrows() if any([answer_in_last_sentence(row.solution,row.answer),parse_boxed_value(row.solution,row.answer)])])/len(df_result)*100
        for i in range(len(df_result)):
            checks.append(check_func(any([answer_in_last_sentence(df_result.loc[i, "solution"],df_result.loc[i, "answer"]),parse_boxed_value(df_result.loc[i, "solution"],df_result.loc[i, "answer"])])))
    elif d == "MMMLU":
        score = sum([1 for _,row in df_result.iterrows() if any([parse_mcqa_value(row.question,row.solution,row.answer)])])/len(df_result)*100
        for i in range(len(df_result)):
            checks.append(check_func(parse_mcqa_value(df_result.loc[i, "question"],df_result.loc[i, "solution"],df_result.loc[i, "answer"])))
    elif d == "KSM":
        score = sum([1 for _,row in df_result.iterrows() if parse_ksm_value(row.question,row.solution,row.answer)])/len(df_result)*100
        for i in range(len(df_result)):
            checks.append(check_func(parse_ksm_value(df_result.loc[i, "question"],df_result.loc[i, "solution"],df_result.loc[i, "answer"])))
    scores[d] = score
    df_result["check"] = checks
    df_result.to_csv(f"{lang}_check_results/{model_name.replace('/', '_')}/{d}_check.csv")


os.makedirs(f"{lang}_check_json_result", exist_ok=True)
with open(f"{lang}_check_json_result/{model_name.replace('/', '_')}.json", "w") as f:
    json.dump(scores, f, indent=4)

print(f'########### {model_name} ###########')
for k,v in scores.items():
    print(k, v)