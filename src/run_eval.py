import os
import argparse
from models import load_vllm_model, litellm_models
from data import generate_queries_local, generate_queries_litellm, parse_boxed_value, answer_in_last_sentence
from datasets import load_dataset
import pandas as pd
from litellm import batch_completion
from tqdm import tqdm
import huggingface_hub

huggingface_hub.login("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")

# Set up argparse
parser = argparse.ArgumentParser(description="Generate predictions and save results to CSV.")
parser.add_argument('--cats', nargs='+', default=['MATH', 'GSM8K', 'OMNI_MATH', "MMMLU", "KSM"],
                    help="List of dataset categories to process, separated by spaces.")
parser.add_argument('--model_name', type=str, default='gpt-4o',
                    help="Name of the model to use for generating predictions.")
args = parser.parse_args()

# Retrieve arguments
cats = args.cats
model_name = args.model_name

# Load datasets
dfs = {cat: pd.DataFrame(load_dataset('HAERAE-HUB/ksm', cat)['test']) for cat in cats}

# Load model
if model_name not in litellm_models:
    llm, params = load_vllm_model(model_name)

if not os.path.exists('results'):
    os.makedirs('results')
model_path = model_name.replace('/','_')
if not os.path.exists(f'results/{model_path}'):
    os.makedirs(f'results/{model_path}')


# Process each dataset and generate outputs
scores = {}
for k, df in tqdm(dfs.items(),total=len(dfs)):
    if model_name in litellm_models:
        prompts = generate_queries_litellm(df, model_name)
        responses = batch_completion(model=model_name, messages = prompts)
        outputs = [resp.choices[0].message.content for resp in responses]
    else:
        prompts = generate_queries_local(df, model_name)
        outputs = llm.generate(prompts, params)
        outputs = [output.outputs[0].text for output in outputs]
    
    df_result = pd.DataFrame({'question': df.question, 'answer': df.answer, 'solution': outputs})
    score = sum([1 for _,row in df_result.iterrows() if any([answer_in_last_sentence(row.solution,row.answer),parse_boxed_value(row.solution,row.answer)])])/len(df)*100
    scores[k] = score
    df_result.to_csv(f"results/{model_path}/{k}.csv", index=False)

print(f'########### {model_name} ###########')
for k,v in scores.items():
    print(k, v)
