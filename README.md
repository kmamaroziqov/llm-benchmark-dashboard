# LLM Benchmark for Uzbek Intent Extraction

This project benchmarks several local LLMs running via Ollama on a specific task: extracting structured JSON intents from Uzbek user requests (e.g., "Gaz ga 30 ming pul tashla").

## Features
- **Time Metrics**: Measures load time, pre-processing (prompt evaluation) time, and prediction (evaluation) time.
- **Performance Metrics**: Calculates accuracy based on exact key-value matching of the expected JSON output.
- **Memory Metrics**: Tracks the model's memory footprint (RAM/VRAM) using `ollama ps` and system-wide VRAM/RAM usage.
- **Analytics Dashboard**: A Streamlit app to visualize and compare the results.

## Setup

1. Ensure you have [Ollama](https://ollama.com/) installed and running.
2. Ensure you have pulled the required models:
   ```bash
   ollama pull alloma-8b-q4:latest
   ollama pull gemma3:12b
   ollama pull gemma3:4b
   ollama pull deepseek-r1:8b
   ollama pull qwen3:8b
   ollama pull qwen3:4b
   ```
3. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Benchmark

1. Run the benchmark script. This will iterate through all models and prompts, saving the results in the `results/` directory as `.jsonl` files.
   ```bash
   python benchmark.py
   ```

## Viewing the Dashboard

1. Start the Streamlit dashboard to view the analytics:
   ```bash
   streamlit run dashboard.py
   ```
2. Open the provided local URL in your browser to see the comparison charts.