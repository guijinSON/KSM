import torch
from transformers import AutoModel, AutoTokenizer
from openai import OpenAI
model_name = "Qwen/Qwen2.5-Math-RM-72B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8081/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

import pandas as pd
from tqdm import tqdm

df = pd.read_csv('OWM-3M-filtered.csv').dropna()

output = []

for _,row in tqdm(df.iterrows(),total=len(df)):
    chat = [
    {"role": "system", "content": "Solve the given question.\nAfter solving the problem, state your final answer in the one of the following format: $\\boxed{N}$."},
    {"role": "user", "content": row.translated_question},
    {"role": "assistant", "content": row.translated_solution}
]
    conversation_str = tokenizer.apply_chat_template(
                chat, 
                tokenize=False, 
                add_generation_prompt=False
            )
    output.append(conversation_str)

df['message'] = output

from more_itertools import batched
import time
reward_list = []
for _,row in tqdm(df.iterrows(),total=len(df)):
    try:
        responses = client.embeddings.create(
                        input= [row.message],
                        model=model,
                    )
        reward = responses.data[0].embedding[-1]
    except:
        reward = -9999
    reward_list.append(reward)
    if _ % 1000 == 0:
        pd.DataFrame(reward_list).to_csv('log.csv',index=False)