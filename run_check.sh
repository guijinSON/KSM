#!/bin/bash

# List of model names to iterate over
models=(
    # "gpt-4o-mini-2024-07-18"
    # "models/meta-llama_Llama-3.1-8B-Instruct"
    # "models/CohereForAI_aya-expanse-8b"
    # "models/Qwen_Qwen2.5-7B-Instruct"
    # "models/Qwen_Qwen2.5-Math-7B-Instruct"
    # "models/nvidia_OpenMath2-Llama3.1-8B"
    # "models/deepseek-ai_deepseek-math-7b-instruct"
    # "models/amphora_eli"
    # "models/google_gemma-2-9b-it"
    # "models/CohereForAI_aya-expanse-32b"
    # "models/Qwen_Qwen2.5-7B-Instruct"
    # "models/Qwen_Qwen2.5-14B-Instruct"
    # "models/Qwen_Qwen2.5-32B-Instruct"
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
    "Qwen/Qwen2.5-32B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.1-70B-Instruct"
    "gpt-4o-mini"
    "gpt-4o"
    )

# export HF_TOKEN="YOURHF_TOKEN"
# export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

# Loop through each model and run the Python script
for model_name in "${models[@]}"; do
  echo "Running check for model: $model_name"
  python src/check.py --model_name "$model_name" --lang_type "ko"
done
