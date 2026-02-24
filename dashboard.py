import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px

RESULTS_DIR = "results"

st.set_page_config(page_title="LLM Benchmark Dashboard", layout="wide")

st.title("LLM Benchmark Dashboard: Intent Extraction (Uzbek)")

@st.cache_data
def load_data():
    data = []
    if not os.path.exists(RESULTS_DIR):
        return pd.DataFrame()
        
    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".jsonl")]
    if not files:
        return pd.DataFrame()

    for filename in files:
        filepath = os.path.join(RESULTS_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        record = json.loads(line)
                        data.append(record)
                    except json.JSONDecodeError:
                        pass
        except Exception:
             pass
            
    return pd.DataFrame(data)

df = load_data()

if df.empty:
    st.warning("No benchmark results found. Please run `python benchmark.py` first.")
    st.stop()

# Aggregate metrics
agg_df = df.groupby('model').agg(
    avg_accuracy=('accuracy', 'mean'),
    avg_load_time=('load_time_sec', 'mean'),
    avg_prompt_eval_time=('prompt_eval_time_sec', 'mean'),
    avg_eval_time=('eval_time_sec', 'mean'),
    model_memory_gb=('model_memory_gb', 'max'),
    sys_vram_gb=('sys_vram_gb', 'max')
).reset_index()

# Convert accuracy to percentage
agg_df['avg_accuracy'] = agg_df['avg_accuracy'] * 100

st.header("Summary Metrics")
st.dataframe(agg_df.style.format({
    'avg_accuracy': '{:.1f}%',
    'avg_load_time': '{:.2f}s',
    'avg_prompt_eval_time': '{:.2f}s',
    'avg_eval_time': '{:.2f}s',
    'model_memory_gb': '{:.2f} GB',
    'sys_vram_gb': '{:.2f} GB'
}))

col1, col2 = st.columns(2)

with col1:
    st.subheader("Performance: Accuracy (%)")
    fig_acc = px.bar(agg_df, x='model', y='avg_accuracy', color='model', text_auto='.1f',
                     labels={'avg_accuracy': 'Accuracy (%)', 'model': 'Model'})
    st.plotly_chart(fig_acc, use_container_width=True)

with col2:
    st.subheader("Time: Prediction Time (Eval Duration)")
    fig_time = px.bar(agg_df, x='model', y='avg_eval_time', color='model', text_auto='.2f',
                      labels={'avg_eval_time': 'Eval Time (seconds)', 'model': 'Model'})
    st.plotly_chart(fig_time, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Time: Pre-processing Time (Prompt Eval)")
    fig_prompt = px.bar(agg_df, x='model', y='avg_prompt_eval_time', color='model', text_auto='.2f',
                        labels={'avg_prompt_eval_time': 'Prompt Eval Time (seconds)', 'model': 'Model'})
    st.plotly_chart(fig_prompt, use_container_width=True)

with col4:
    st.subheader("Memory: Model Size in RAM/VRAM (GB)")
    fig_mem = px.bar(agg_df, x='model', y='model_memory_gb', color='model', text_auto='.2f',
                     labels={'model_memory_gb': 'Memory (GB)', 'model': 'Model'})
    st.plotly_chart(fig_mem, use_container_width=True)

st.header("Detailed Results")
st.dataframe(df[['model', 'prompt', 'expected', 'actual', 'accuracy', 'eval_time_sec']])
