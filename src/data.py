from transformers import AutoTokenizer
import re
from collections import Counter
from jinja2.exceptions import TemplateError
from latex2sympy2 import latex2sympy
from sympy import latex, simplify
from prompts import prompts

system_message = " "

def check_duplication(input_str, thres=10):
    count = dict(Counter(input_str.split()))
    dup_w_count = sum([1 for k,v in count.items() if v >30])
    return dup_w_count > thres
    
def check_sentence_num(source_str, target_str, thres=10):
    source_s_count = len(source_str.split('.'))
    target_s_count = len(target_str.split('.'))
    return abs(target_s_count-source_s_count) > thres

def check_answer_len(source_str, target_str, thres=1000):
    return abs(len(source_str) - len(target_str)) > thres

def max_duplicated(input_str, thres=50):
    count = dict(Counter(input_str.split()))
    return any([True for k,v in count.items() if v > thres])

def match_digit_num(source_str, target_str, thres=10):
    source_n_count = len([1 for _ in source_str if _.isdigit()])
    target_n_count = len([1 for _ in target_str if _.isdigit()])
    return abs(target_n_count-source_n_count) > three

def latex_expressions_equal(solution: str, answer: str) -> str:
    try:
        sympy_solution, sympy_answer = latex2sympy(solution), latex2sympy(answer)
        simplified_solution, simplified_answer = simplify(sympy_solution), simplify(sympy_answer)
        return latex(simplified_solution) == latex(simplified_answer)
    except:
        return False
    return latex(simplified_expr)

def mcqa_formatting(question, answer):
    number_list = ["\n1. ", "\n2. ", "\n3. ", "\n4. ", "\n5. "]
    check = {num[1]: question.split(num)[-1].split("\n")[0].strip() for num in number_list if num in question}
    
    try:
        answer_choice = float(answer)
        answer_content = check[str(answer)]
    except:
        answer_content = answer
        for k, v in check.items():
            if v == answer_content:
                answer_choice = float(k)

    return answer_choice, answer_content
    
def answer_in_last_sentence(input_string, answer):
    last_sentence = str(input_string).strip().split('\n')[-1]
    numbers_in_last_sentence = [float(num) for num in re.findall(r'\d+\.?\d*', last_sentence)]
    if len(numbers_in_last_sentence) > 0:
        return float(answer) == numbers_in_last_sentence[-1]
    else:
        return False
    
def convert_to_int_safe(string_number):
    cleaned = str(string_number).replace(',', '')
    if re.fullmatch(r'\d+(\.\d+)?', cleaned): 
        return float(cleaned)
    else:
        return string_number
        
def parse_boxed_value(text,answer):
    match = re.search(r'\\boxed\{\s*([\d,]+(?:\.\d+)?)\s*\}', str(text))
    if match:
        pred = convert_to_int_safe(match.group(1))
        c1 = float(pred) == answer
        c2 = str(pred) == str(answer)
        return any([c1,c2])
    return False

def parse_boxed_content_value(text, answer):
    match = re.search(r'\\boxed\{([a-zA-Z0-9가-힣\=\+\-\*/\^\_\(\)\{\}\[\]\\ ]+)\}', str(text))
    if match:
        return any([str(answer) == str(match.group(1)), latex_expressions_equal(str(match.group(1)), str(answer))])
    return False

def parse_mcqa_value(question, text, answer):
    answer_choice, answer_content = mcqa_formatting(question, answer)

    return any([answer_in_last_sentence(text, answer_choice), parse_boxed_value(text, answer_choice), parse_boxed_content_value(text, answer_content)])
                
def parse_ksm_value(question,text,answer):
    if ("1." in question) and ("2." in question) and ("3." in question) and ("4." in question):
        return parse_mcqa_value(question, text, answer)
    else:
        try:
            answer = float(answer)
            return any([answer_in_last_sentence(text, answer), parse_boxed_value(text, answer)])
        except:
            return parse_boxed_content_value(text, answer)
        
def generate_queries_local(df, model_name, prompt_id):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qrys = []
    
    for _,row in df.iterrows():
        if prompt_id in ["en",'oasst_en', 'e2e', 'e2k']:
            text = row.original
        else:
            text = row.question

        if 'oasst' in prompt_id:
            qry = prompts[prompt_id].replace("{instruction}",text)
        else:
            if prompt_id in ["k2k", "e2k"]:
                msg = prompts["ko"]
            elif prompt_id in ["e2e", "k2e"]:
                msg = prompts["en"]
            else:
                msg = prompts[prompt_id]

        try:
            messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": " ".join([text, msg])}
                ]
            qry = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
        except TemplateError as e:
            if str(e) == 'System role not supported':
                messages = [
                    {"role": "user", "content": f"{system_message}" + '\n\n'+ text + msg}
                ]
                qry = tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                print(f"An error occurred: {e}")
        qrys.append(qry)
    return qrys

def generate_queries_litellm(df, model_name, prompt_id):
    qrys = []
    for _,row in df.iterrows():
        if prompt_id == "en":
            text = row.original
        else:
            text = row.question
        messages =  [
            {"role": "system","content": system_message + '\n\n' + prompts[prompt_id]},       
            {"role": "user","content": text}
        ]
        qrys.append(messages)
    return qrys
