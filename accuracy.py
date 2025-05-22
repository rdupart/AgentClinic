import json
import openai
import os
import re

openai.api_key = "YOUR_OPENAI_API_KEY"  # set your API key here or from env var

def query_model(backend, prompt, system_prompt="", **kwargs):
    # Simplified query model function for GPT-4 backend
    response = openai.ChatCompletion.create(
        model="gpt-4.1-mini" if backend == "gpt4" else backend,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        **kwargs
    )
    return response['choices'][0]['message']['content']

def semantic_compare_diagnoses(pred_diag, gold_diag, backend="gpt4"):
    prompt = (
        f"Here is the correct diagnosis: {gold_diag}\n"
        f"Here was the doctor's diagnosis: {pred_diag}\n"
        "Are these referring to the same underlying medical condition? Please respond only with Yes or No."
    )
    system_prompt = (
        "You are an expert medical evaluator. Determine if the provided doctor's diagnosis matches "
        "the correct diagnosis in meaning, even if phrased differently. Respond only with 'Yes' or 'No'."
    )
    response = query_model(backend, prompt, system_prompt=system_prompt)
    return response.strip().lower() == "yes"

def load_output_file(filepath):
    # Adjust this depending on your output format!
    # For example, if each output JSON contains:
    # { "predicted_diagnosis": "...", "correct_diagnosis": "..." }
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("predicted_diagnosis", ""), data.get("correct_diagnosis", "")

def evaluate_outputs(folder_path, backend="gpt4"):
    files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    total = len(files)
    correct = 0

    for i, file in enumerate(files, 1):
        pred_diag, correct_diag = load_output_file(os.path.join(folder_path, file))
        if not pred_diag or not correct_diag:
            print(f"[WARN] Missing data in {file}, skipping...")
            continue
        
        match = semantic_compare_diagnoses(pred_diag, correct_diag, backend=backend)
        print(f"[{i}/{total}] File: {file} - Match: {'Yes' if match else 'No'}")
        if match:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"\nFinal Semantic Accuracy: {accuracy:.2%} ({correct}/{total})")

if __name__ == "__main__":
    folder_with_outputs = "path/to/your/output/files"
    evaluate_outputs(folder_with_outputs)
