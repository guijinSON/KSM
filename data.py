from transformers import AutoTokenizer
import torch
import re
from jinja2.exceptions import TemplateError

system_message = """Solve the given question.
After solving the problem, state your final answer in the one of the following format: $\\boxed{N}$."""

def answer_in_last_sentence(input_string, answer):
    last_sentence = str(input_string).strip().split('\n')[-1]
    numbers_in_last_sentence = [float(num) for num in re.findall(r'\d+\.?\d*', last_sentence)]
    return answer in numbers_in_last_sentence
    
def parse_boxed_value(text,answer):
    match = re.search(r'\\boxed\{(\d+)\}', str(text))
    if match:
        return float(match.group(1)) == answer
    return False
        
def generate_queries_local(df, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qrys = []
    
    for _,row in df.iterrows():
        try:
            messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": row.question}
                ]
            qry = tokenizer.apply_chat_template(messages, tokenize=False)
            
        except TemplateError as e:
            if str(e) == 'System role not supported':
                messages = [
                    {"role": "user", "content": system_message + '\n\n'+ row.question}
                ]
                qry = tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                print(f"An error occurred: {e}")
        qrys.append(qry)
    return qrys

def generate_queries_litellm(df,model_name):
    qrys = []
    for _,row in df.iterrows():
        messages =  [
            {"role": "system","content": system_message},       
            {"role": "user","content": row.question}
        ]
        qrys.append(messages)
    return qrys