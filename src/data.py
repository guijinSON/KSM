from transformers import AutoTokenizer
import re
from collections import Counter
from jinja2.exceptions import TemplateError
from sympy import sympify, simplify
from sympy.parsing.latex import parse_latex

system_message = """Solve the given question.
After solving the problem, state your final answer in the following format: $\\boxed{N}$."""

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
    return abs(target_n_count-source_n_count) > thres

def latex_expressions_equal(solution, answer):
    try:
        sympy_expr1 = parse_latex(solution)
        sympy_expr2 = parse_latex(answer)
        return sympy_expr1 == sympy_expr2
    except:
        return False

def mcqa_formatting(question, answer):
    number_list = ["1.", "2.", "3.", "4.", "5."]
    check = {num[0]: question.split(num)[-1].split("\n")[0].strip() for num in number_list if num in question}
    
    try:
        float(answer)
        answer_choice = str(answer)
        answer_content = check[str(answer_choice)]
    except:
        answer_content = answer
        for k, v in check.items():
            if v == answer_content:
                answer_choice = k

    return answer_choice, answer_content
    
def answer_in_last_sentence(input_string, answer):
    last_sentence = str(input_string).strip().split('\n')[-1]
    numbers_in_last_sentence = [float(num) for num in re.findall(r'\d+\.?\d*', last_sentence)]
    if len(numbers_in_last_sentence) > 0:
        return float(answer) == numbers_in_last_sentence[-1]
    else:
        return False
    
def parse_boxed_value(text,answer):
    match = re.search(r'\\boxed\{(\d+)\}', str(text))
    if match:
        c1 = float(match.group(1)) == answer
        c2 = str(match.group(1)) == str(answer)
        return any([c1,c2])
    return False

def parse_boxed_content_value(text, answer):
    match = re.search(r'\\boxed\{([a-zA-Z0-9가-힣\=\+\-\*/\^\_\(\)\{\}\[\]\\ ]+)\}', str(text))
    if match:
        return any([str(answer) == str(match.group(1)), latex_expressions_equal(match.group(1), answer)])
    return False

def parse_mcqa_value(question, text, answer):
    answer_choice, answer_content = mcqa_formatting(question, answer)

    return any([answer_in_last_sentence(text, answer_choice), parse_boxed_value(text, answer_choice), parse_boxed_content_value(text, answer_content)])
                
def parse_ksm_value(question,text,answer):
    if ("1." in question) and ("2." in question) and ("3." in question) and ("4." in question):
        return parse_mcqa_value(question, text, answer)
    else:
        try:
            float(answer)
            return any([answer_in_last_sentence(text, answer), parse_boxed_value(text, answer)])
        except:
            return parse_boxed_content_value(text, answer)
        
def generate_queries_local(df, model_name, lang):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qrys = []
    
    for _,row in df.iterrows():
        if lang == "ko":
            text = row.question
        elif lang == "en":
            text = row.original
        try:
            messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": text}
                ]
            qry = tokenizer.apply_chat_template(messages, tokenize=False)
            
        except TemplateError as e:
            if str(e) == 'System role not supported':
                messages = [
                    {"role": "user", "content": system_message + '\n\n'+ text}
                ]
                qry = tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                print(f"An error occurred: {e}")
        qrys.append(qry)
    return qrys

def generate_queries_litellm(df, model_name, lang):
    qrys = []
    for _,row in df.iterrows():
        if lang == "ko":
            text = row.question
        elif lang == "en":
            text = row.original
        messages =  [
            {"role": "system","content": system_message},       
            {"role": "user","content": text}
        ]
        qrys.append(messages)
    return qrys
