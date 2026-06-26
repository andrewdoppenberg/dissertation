import os
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# track-records_sim.py has a hyphen in its name, so it can't be imported with a
# normal `import`; load it explicitly via importlib instead.
_sim_path = Path(__file__).resolve().parent / "track-records_sim.py"
_spec = importlib.util.spec_from_file_location("track_records_sim", _sim_path)
track_records_sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(track_records_sim)
TRModel = track_records_sim.TRModel
PublicExamModel = track_records_sim.PublicExamModel

# -------------------------------------------------------------
# Configuration Constants
# -------------------------------------------------------------
N_STEPS = 50                # Simulate fewer steps to catch early anomaly effects
N_AGENTS = 30               # Standard group size
ANOMALY_THRESHOLD = 0.05    # Minimum deviation considered anomalous
HYP_SPACE = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) # Available hypotheses

# Write figures next to the other chapter figures
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def _anomaly_brier_worker(model_type):
    """
    Worker task to simulate a single run and measure the correlation 
    between an anomalous track record and ultimate agent accuracy (Brier score).
    """
    # Include 0.0 and 1.0 as potential truths
    truth = np.random.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    truth_idx = np.argmin(np.abs(HYP_SPACE - truth))
    
    if model_type == 'baseline':
        model = TRModel(truth, feedback_rate=1, n_agents=N_AGENTS)
    else:
        model = PublicExamModel(truth, feedback_rate=1, n_agents=N_AGENTS)
    
    # Run standard accumulation
    for _ in range(N_STEPS):
        model.step()
        
    anomalous_count = 0
    brier_t_scores = []
    
    # Evaluate every agent post-run
    for agent in model.schedule.agents:
        if len(agent.tosses) == 0:
            continue
            
        # Agent perceived an objective event frequency notably separated from truth
        obs_freq = np.mean(agent.tosses)
        if abs(obs_freq - truth) >= ANOMALY_THRESHOLD:
            anomalous_count += 1
            
        # Categorical Brier Score (BrierT)
        brier_t = np.sum(agent.cred**2) + 1 - 2 * agent.cred[truth_idx]
        brier_t_scores.append(brier_t)
            
    return anomalous_count, np.mean(brier_t_scores)

def run_parallel_experiments(model_type, n_runs=1000):
    """
    Executes _anomaly_brier_worker in parallel to scale over CPUs.
    """
    cores = cpu_count()
    tasks = [model_type] * n_runs
    with Pool(processes=cores) as pool:
        results = list(tqdm(pool.imap_unordered(_anomaly_brier_worker, tasks), total=n_runs, desc=f"Running {model_type}"))
    return results

if __name__ == "__main__":
    N_RUNS = 10000
    # Global formatting options
    plt.rcParams['font.family'] = 'Garamond'
    
    # -------------------------------------------------------------
    # 1. Parallel executions
    # -------------------------------------------------------------
    baseline_results = run_parallel_experiments('baseline', n_runs=N_RUNS)
    public_exam_results = run_parallel_experiments('public_exam', n_runs=N_RUNS)
    
    base_counts, base_briers = zip(*baseline_results)
    pub_counts, pub_briers = zip(*public_exam_results)
    
    # Aggregate into DataFrames specifically shaped for Seaborn visualizations
    df_all = pd.concat([
        pd.DataFrame({'Anomalous_Count': base_counts, 'BrierT': base_briers, 'Model': 'track records'}),
        pd.DataFrame({'Anomalous_Count': pub_counts, 'BrierT': pub_briers, 'Model': 'communal'})
    ])
    
    sns.set_theme(style="whitegrid", rc={"font.family": "Garamond"})
    
    # Explicit color mapping and consistent order
    color_map = {'track records': 'red', 'communal': 'blue'}
    consistent_order = ['track records', 'communal'] # Define the order once
    
    # --- PLOT 1: Histogram ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    bins = np.arange(-0.5, N_AGENTS + 1.5, 1)
    
    # Use hue_order instead of pd.Categorical
    sns.histplot(data=df_all, x='Anomalous_Count', hue='Model', palette=color_map,
                 hue_order=consistent_order, stat="probability", common_norm=False,linewidth=0, alpha=0.5, bins=bins, ax=ax1, multiple="dodge")
    
    ax1.set_xlabel('Number of Anomalous Agents', fontsize=18)
    ax1.set_ylabel('Frequency', fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    
    # Format Seaborn's native legend
    leg1 = ax1.get_legend()
    if leg1:
        leg1.set_title(None)
        for text in leg1.get_texts():
            text.set_fontsize(16)
            
    fig1.tight_layout()
    fig1.savefig(os.path.join(FIGURES_DIR, 'anomaly_distribution.pdf'), format='pdf', bbox_inches='tight')
    
    # --- PLOT 2: Epistemic Accuracy vs Anomalous Count ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    # Use hue_order instead of pd.Categorical
    sns.lineplot(data=df_all, x='Anomalous_Count', y='BrierT', hue='Model', 
                 palette=color_map, hue_order=consistent_order, marker='o', 
                 errorbar=None, ax=ax2)
        
    ax2.set_xlabel('Number of Anomalous Agents', fontsize=18)
    ax2.set_ylabel('Average Brier Score', fontsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    
    # Format Seaborn's native legend
    leg2 = ax2.get_legend()
    if leg2:
        leg2.set_title(None)
        for text in leg2.get_texts():
            text.set_fontsize(16)
            
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIGURES_DIR, 'anomaly_accuracy.pdf'), format='pdf', bbox_inches='tight')
    
    plt.show()