import torch
from transformers import AutoModel, AutoTokenizer
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from more_itertools import batched
import time

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

reward_list = []
idx = 0
batch_size=10000
batches = [df[i:i + batch_size] for i in range(0, df.shape[0], batch_size)]
for batch in batches:
    rewards = []
    for _,row in tqdm(batch.iterrows(),total=10000):
        try:
            responses = client.embeddings.create(
                            input= [row.message],
                            model=model,
                        )
            reward = responses.data[0].embedding[-1]
        except:
            reward = -9999
        rewards.append(reward)
    batch['reward'] = rewards
    batch.to_csv(f'data/log-{idx}.csv',index=False)
    idx+=1
