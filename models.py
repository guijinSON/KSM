from vllm import LLM, SamplingParams
import torch

litellm_models = [
    'gpt-4o',
    'gpt-4o-mini-2024-07-18'
]
    
def load_vllm_model(model_name):
    llm = LLM(model_name, tensor_parallel_size=torch.cuda.device_count(),max_model_len=8192)
    params = SamplingParams(temperature=0.8, top_p=0.95, min_tokens=8, max_tokens=2048)
    return llm, params