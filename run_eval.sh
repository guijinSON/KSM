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
    "models/Qwen_Qwen2.5-7B-Instruct"
    "models/Qwen_Qwen2.5-14B-Instruct"
    "models/Qwen_Qwen2.5-32B-Instruct"
    )



# Loop through each model and run the Python script
for model_name in "${models[@]}"; do
  echo "Running evaluation for model: $model_name"
  python src/run_eval.py --cats GSM8K MATH OMNI_MATH --model_name "$model_name"
done
