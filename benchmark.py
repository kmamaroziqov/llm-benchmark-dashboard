import json
import time
import requests
import subprocess
import os
import psutil

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODELS = [
    "kmamaroziqov/alloma-8b-q4:latest",
    "gemma3:12b",
    "gemma3:4b",
    "deepseek-r1:8b",
    "qwen3:8b",
    "qwen3:4b",
    "llama3.1:8b"
]
DATASET_FILE = "dataset.json"
RESULTS_DIR = "results"

SYSTEM_PROMPT = """You are a financial assistant API. Your task is to extract information from user requests in Uzbek and output ONLY a valid JSON object.

Allowed Values:
- Intents: 'pay', 'transfer', 'check_balance'
- Categories (only for 'pay' intent): 'utilities', 'mobile', 'internet', 'debt'
- Subcategories: 'gas', 'electricity', 'water', 'wifi'
- Currencies: 'UZS', 'USD'

Extraction Rules:
1. Intent:
   - 'transfer': Sending money to a person, relative, or card. Keywords: "o'tkaz", "tashla" (if to person). Do NOT set a 'category' or 'subcategory'.
   - 'pay': Paying for a service (phone, light, gas, water, internet). Keywords: "to'la", "sol", "tashla" (if to service). MUST set a 'category'.
   - 'check_balance': Checking account balance. No other fields needed.
2. Category (for 'pay'):
   - 'utilities': For gas, electricity (svet), water, trash. Even if the word "qarz" (debt) is used for these, use 'utilities'.
   - 'mobile': For phone payment. Do NOT add a subcategory.
   - 'internet': For internet/wifi.
   - 'debt': For paying back a general debt (qarz) NOT related to utilities.
3. Subcategory:
   - Specific service name in English: 'gas', 'electricity', 'water', 'wifi'.
   - Only for 'utilities' or 'internet' categories.
   - For 'electricity': maps from 'svet', 'elektr', 'tok'.
   - For 'gas': maps from 'gaz'.
   - For 'water': maps from 'suv'.
4. Amount:
   - Convert words to integer numbers (e.g., "30 ming" -> 30000, "1 million" -> 1000000).
   - "ming" = * 1000.
5. Currency:
   - DEFAULT to 'UZS' if not specified or if "so'm" is used.
   - Only use 'USD' if explicitly mentioned.
6. Target:
   - REQUIRED for 'transfer' intent. The exact name of the recipient WITHOUT suffixes like "-ga", "-ni", "-ning".
   - e.g., "Oyimga" -> "Oyim", "Akamning kartasiga" -> "Akamning kartasi".

Examples:
User: "Gaz ga 30 ming pul tashla"
JSON: {"intent": "pay", "category": "utilities", "subcategory": "gas", "amount": 30000, "currency": "UZS"}

User: "Oyimga 500 ming so'm o'tkazib yubor"
JSON: {"intent": "transfer", "amount": 500000, "currency": "UZS", "target": "Oyim"}

User: "Svetdan qarzdorlikni to'lash uchun 100000 so'm"
JSON: {"intent": "pay", "category": "utilities", "subcategory": "electricity", "amount": 100000, "currency": "UZS"}

User: "Telefonimga 10 ming sol"
JSON: {"intent": "pay", "category": "mobile", "amount": 10000, "currency": "UZS"}

Output ONLY the JSON object. No markdown, no explanations."""

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_dataset():
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_ollama_memory(model_name):
    """Get memory usage of the loaded model from ollama ps (in GB)"""
    try:
        res = subprocess.check_output(['ollama', 'ps'], encoding='utf-8')
        lines = res.strip().split('\n')[1:]
        for line in lines:
            parts = line.split()
            if len(parts) >= 4 and parts[0] == model_name:
                size = float(parts[2])
                unit = parts[3]
                if unit == 'GB': return size
                if unit == 'MB': return size / 1024
    except Exception as e:
        print(f"Error getting memory for {model_name}: {e}")
    return 0.0

def get_system_vram():
    """Get total VRAM used via nvidia-smi (in GB)"""
    try:
        res = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], encoding='utf-8')
        return sum([int(x) for x in res.strip().split('\n')]) / 1024
    except:
        return 0.0

def calculate_accuracy(expected, actual_json_str):
    """Calculate a simple accuracy score based on key-value matching."""
    try:
        # Some models might wrap in markdown despite instructions
        actual_json_str = actual_json_str.strip()
        if actual_json_str.startswith("```json"):
            actual_json_str = actual_json_str[7:]
        if actual_json_str.startswith("```"):
            actual_json_str = actual_json_str[3:]
        if actual_json_str.endswith("```"):
            actual_json_str = actual_json_str[:-3]
            
        actual = json.loads(actual_json_str)
        score = 0
        total = len(expected)
        
        for k, v in expected.items():
            if k in actual and str(actual[k]).lower() == str(v).lower():
                score += 1
                
        return score / total if total > 0 else 0
    except json.JSONDecodeError:
        return 0.0

def run_benchmark():
    ensure_dir(RESULTS_DIR)
    dataset = load_dataset()
    
    for model in MODELS:
        print(f"\\n{'='*40}\\nBenchmarking model: {model}\\n{'='*40}")
        
        # Pre-load the model to get accurate memory readings
        print(f"Pre-loading {model}...")
        requests.post(OLLAMA_URL, json={"model": model, "prompt": "test", "stream": False, "options": {"num_gpu": 99}})
        time.sleep(2) # Give it a moment to settle
        
        model_memory_gb = get_ollama_memory(model)
        sys_vram_gb = get_system_vram()
        sys_ram_gb = psutil.virtual_memory().used / (1024**3)
        
        print(f"Model Memory (Ollama): {model_memory_gb:.2f} GB")
        print(f"System VRAM Used: {sys_vram_gb:.2f} GB")
        print(f"System RAM Used: {sys_ram_gb:.2f} GB")
        
        results_file = os.path.join(RESULTS_DIR, f"results_{model.replace(':', '_').replace('/', '_')}.jsonl")
        
        with open(results_file, 'w', encoding='utf-8') as f:
            for i, item in enumerate(dataset):
                prompt = item['prompt']
                expected = item['expected']
                
                full_prompt = f"{SYSTEM_PROMPT}\\n\\nUser Request: {prompt}\\nJSON Output:"
                
                payload = {
                    "model": model,
                    "prompt": full_prompt,
                    "format": "json",
                    "stream": False,
                    "options": {
                        "temperature": 0.0, # Greedy decoding for consistent JSON
                        "num_gpu": 99 # Ensure all layers are offloaded to GPU
                    }
                }
                
                print(f"  [{i+1}/{len(dataset)}] Testing prompt: '{prompt}'")
                
                try:
                    response = requests.post(OLLAMA_URL, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Ollama returns durations in nanoseconds
                    load_time = data.get('load_duration', 0) / 1e9
                    prompt_eval_time = data.get('prompt_eval_duration', 0) / 1e9
                    eval_time = data.get('eval_duration', 0) / 1e9
                    
                    actual_output = data.get('response', '{}')
                    accuracy = calculate_accuracy(expected, actual_output)
                    
                    result_record = {
                        "model": model,
                        "prompt": prompt,
                        "expected": expected,
                        "actual": actual_output,
                        "accuracy": accuracy,
                        "load_time_sec": load_time,
                        "prompt_eval_time_sec": prompt_eval_time,
                        "eval_time_sec": eval_time,
                        "model_memory_gb": model_memory_gb,
                        "sys_vram_gb": sys_vram_gb,
                        "sys_ram_gb": sys_ram_gb
                    }
                    
                    f.write(json.dumps(result_record, ensure_ascii=False) + '\n')
                    print(f"    -> Accuracy: {accuracy*100:.0f}% | Eval Time: {eval_time:.2f}s")
                    
                except Exception as e:
                    print(f"    -> Error: {e}")
                    
        # Unload the model to free up memory for the next one
        print(f"Unloading {model}...")
        requests.post(OLLAMA_URL, json={"model": model, "keep_alive": 0})
        time.sleep(3)

if __name__ == "__main__":
    run_benchmark()
    print("\\nBenchmarking complete! Run 'streamlit run dashboard.py' to view the results.")
