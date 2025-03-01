import os
import argparse
from models import load_vllm_model, litellm_models
from data import generate_queries_local, generate_queries_litellm, parse_boxed_value, answer_in_last_sentence, parse_mcqa_value, parse_ksm_value
from datasets import load_dataset
import pandas as pd
from litellm import batch_completion
from tqdm import tqdm
import huggingface_hub
import json

# Set up argparse
parser = argparse.ArgumentParser(description="Generate predictions and save results to CSV.")
parser.add_argument('--cats', nargs='+', default=['MATH', 'GSM8K', 'OMNI_MATH', "MMMLU", "KSM"],
                    help="List of dataset categories to process, separated by spaces.")
parser.add_argument('--model_name', type=str, default='gpt-4o',
                    help="Name of the model to use for generating predictions.")
parser.add_argument('--prompt_id', type=str, default="default",
                    help="Prompt to use for eval.")

args = parser.parse_args()

# Retrieve arguments
cats = args.cats
model_name = args.model_name
prompt_id = args.prompt_id

# Load datasets
dfs = {cat: pd.DataFrame(load_dataset('HAERAE-HUB/HRM8K', cat)['test']) for cat in cats}

# Load model
if model_name not in litellm_models:
    llm, params = load_vllm_model(model_name)

model_path = model_name.replace('/','_')
os.makedirs(f'{prompt_id}_results', exist_ok=True)
os.makedirs(f'{prompt_id}_results/{model_path}', exist_ok=True)

# Process each dataset and generate outputs
scores = {}
for k, df in tqdm(dfs.items(),total=len(dfs)):
    if model_name in litellm_models:
        prompts = generate_queries_litellm(df, model_name, prompt_id)
        responses = batch_completion(model=model_name, messages = prompts)
        outputs = [resp.choices[0].message.content for resp in responses]
    else:
        prompts = generate_queries_local(df, model_name, prompt_id)
        print(prompts[0])
        outputs = llm.generate(prompts, params)
        outputs = [output.outputs[0].text.strip("</s2>") for output in outputs]
    
    df['solution'] = outputs
    df.to_csv(f"{prompt_id}_results/{model_path}/{k}.csv", index=False)
    
    if k in ["GSM8K", "MATH", "OMNI_MATH"]:
        score = sum([1 for _,row in df.iterrows() if any([answer_in_last_sentence(row.solution,row.answer),parse_boxed_value(row.solution,row.answer)])])/len(df)*100
    elif k == "MMMLU":
        score = sum([1 for _,row in df.iterrows() if any([parse_mcqa_value(row.question,row.solution,row.answer)])])/len(df)*100
    elif k == "KSM":
        if prompt_id in ["en", "oasst_en", "e2e", "e2k"]:
            score = sum([1 for _,row in df.iterrows() if parse_ksm_value(row.original,row.solution,row.original_answer)])/len(df)*100
        else:
            score = sum([1 for _,row in df.iterrows() if parse_ksm_value(row.question,row.solution,row.answer)])/len(df)*100
    scores[k] = score
    print(scores)
  
os.makedirs(f"{prompt_id}_json_result", exist_ok=True)
with open(f"{prompt_id}_json_result/{model_path}.json", "w") as f:
    json.dump(scores, f, indent=4)

print(f'########### {model_name} ###########')
for k,v in scores.items():
    print(k, v)
