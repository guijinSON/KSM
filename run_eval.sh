#!/bin/bash

# List of model names to iterate over
models=(
    # "Qwen/Qwen2.5-1.5B-Instruct"
    # "Qwen/Qwen2.5-3B-Instruct"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "Qwen/Qwen2.5-32B-Instruct"
    # "Qwen/Qwen2.5-72B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    # "meta-llama/Llama-3.2-3B-Instruct"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "meta-llama/Llama-3.1-70B-Instruct"
    # "gpt-4o-mini"
    # "gpt-4o"
)

# Set the categories
CATEGORIES="GSM8K MATH OMNI_MATH MMMLU KSM"

# Set the prompt ID
PROMPT_ID="TBST"

# Optional: Set your tokens (uncomment and set as needed)
# export HF_TOKEN="YOUR_HF_TOKEN"
# export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

# Loop through each model and run the Python script
for model_name in "${models[@]}"; do
  echo "Running evaluation for model: $model_name with categories: $CATEGORIES"
  python src/run_eval.py --cats $CATEGORIES --model_name "$model_name" --prompt_id "$PROMPT_ID"
  # Uncomment the next line if you want to clear Hugging Face cache after each run
  # rm -rf ~/.cache/huggingface
done
