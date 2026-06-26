import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Global Plotting Configurations
# ---------------------------------------------------------
# Set global font and styles for all plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Garamond', 'EB Garamond', 'serif']
plt.rcParams['font.size'] = 12

# Standard linestyles to iterate through
LINESTYLES = ['solid', 'dashed', 'dotted', 'dashdot', (0, (3, 1, 1, 1))]

# ---------------------------------------------------------
# 2. Reusable Plotting Function
# ---------------------------------------------------------
def plot_time_series_metric(data_list, labels, param, ylabel, filename, legend_loc='best', invert_y=False):
    """
    Plots a specific metric and exports it as a high-quality PDF.
    Expects data_list to contain pre-aggregated DataFrames where 'Step' is the index.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    
    for idx, df_agg in enumerate(data_list):
        ax.plot(
            df_agg.index, 
            df_agg[param].values, 
            label=labels[idx], 
            alpha=0.7, 
            ls=LINESTYLES[idx % len(LINESTYLES)]
        )

    if invert_y:
        ax.invert_yaxis()
        
    ax.set_xlabel('Time Step')
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc)
    
    plt.tight_layout()
    
    # Export as high-quality PDF
    plt.savefig(f"{filename}.pdf", format='pdf', bbox_inches='tight', dpi=300)
    print(f"Exported: {filename}.pdf")
    
    plt.close(fig)

# ---------------------------------------------------------
# 3. Dataset Group Definitions
# ---------------------------------------------------------
# Single pickle file produced by track-records_sim.py
PKL_PATH = '../results/track-records.pkl'

DATASET_GROUPS = [
    {
        'name': 'huang',
        'keys': [
            'baseline_30',
            'testimony_30',
            'patience_30',
            'less_monopoly_30',
        ],
        'labels': ['baseline', 'testimony', 'patience', 'less monopoly'],
    },
    {
        'name': 'expert_priors',
        'keys': [
            'baseline_30',
            'prior_baseline_30',
            'prior_30',
            'prior_500_baseline_30',
            'prior_500_30',
        ],
        'labels': ['baseline', 'expert baseline (10)', 'expert priors (10)', 'expert baseline (100)', 'expert priors (100)'],
        'plot_overrides': {
            'brier': {
                'keys': [
                    'prior_baseline_30',
                    'prior_30',
                    'prior_500_baseline_30',
                    'prior_500_30',
                ],
                'labels': ['expert baseline (10)', 'expert priors (10)', 'expert baseline (100)', 'expert priors (100)'],
            },
        },
    },
    {
        'name': 'peer_predictions',
        'keys': [
            'baseline_30',
            'peer_prediction_30',
            'prior_baseline_30',
            'prior_peer_prediction_30',
        ],
        'labels': ['baseline', 'peer prediction', 'expert priors baseline', 'expert priors + peer prediction'],
    },
    {
        'name': 'random',
        'keys': [
            'baseline_30',
            'random_30',
            'prior_baseline_30',
            'prior_peer_prediction_random_30',
        ],
        'labels': ['baseline', 'random', 'expert priors baseline', 'expert priors + peer prediction + random'],
        'plot_overrides': {
            'brier': {
                'keys': [
                    'prior_baseline_30',
                    'prior_peer_prediction_random_30',
                ],
                'labels': ['expert priors baseline', 'expert priors + peer prediction + random'],
            },
        },
    },
    {
        'name': 'fuzzy',
        'keys': [
            'baseline_30',
            'testimony_30',
            'random_30',
            'fuzzy_tr_30',
        ],
        'labels': ['baseline', 'testimony', 'random', 'fuzzy'],
    },
    {
        'name': 'delayed',
        'keys': [
            'baseline_30',
            'delayed_30',
        ],
        'labels': ['baseline', 'delayed'],
    },
    {
        'name': 'inertia',
        'keys': [
            'baseline_30',
            'inertial_30',
        ],
        'labels': ['baseline', 'inertia'],
    },
    {
        'name': 'sharing',
        'keys': [
            'sharing_baseline_30',
            'sharing_30',
            'sharing_baseline_5',
            'sharing_5',
        ],
        'labels': ['sharing baseline (N = 30)', 'sharing (N = 30)', 'sharing baseline (N = 5)', 'sharing (N = 5)'],
    },
    {
        'name': 'communal',
        'keys': [
            'baseline_30',
            'communal_30',
        ],
        'labels': ['baseline', 'communal'],
    },
]

# ---------------------------------------------------------
# 4. Plot Configurations
# ---------------------------------------------------------
PLOT_CONFIGS = [
    {
        'param':           'BrierT',
        'ylabel':          'Average Brier',
        'filename_suffix': 'brier',
        'legend_loc':      'lower right',
        'invert_y':        True,
    },
    {
        'param':           'OpinionDiversity',
        'ylabel':          'Opinion Diversity',
        'filename_suffix': 'opinion_diversity',
        'legend_loc':      'right',
        'invert_y':        False,
    },
    {
        'param':           'NeighborDiversity',
        'ylabel':          'Neighbour Diversity',
        'filename_suffix': 'neighbour_diversity',
        'legend_loc':      'upper right',
        'invert_y':        False,
    },
    {
        'param':           'AvgNeighborNoise',
        'ylabel':          'Neighbour Noise',
        'filename_suffix': 'neighbour_noise',
        'legend_loc':      'upper right',
        'invert_y':        True,
    },
]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, '../figures/')
    os.makedirs(figures_dir, exist_ok=True)

    # Load the single pickle produced by track-records_sim.py
    pkl_path = os.path.join(script_dir, PKL_PATH)
    if not os.path.exists(pkl_path):
        print(f"[ERROR] Results file not found: {pkl_path}")
        print("Run track-records_sim.py first to generate track-records.pkl")
        return

    print(f"Loading results from {pkl_path} ...")
    with open(pkl_path, 'rb') as fh:
        results_store = pickle.load(fh)
    print(f"Loaded {len(results_store)} entries: {sorted(results_store.keys())}\n")

    # Process each dataset group and generate plots
    for group in DATASET_GROUPS:
        print(f"=== Group: {group['name']} ===")

        for cfg in PLOT_CONFIGS:
            plot_override = group.get('plot_overrides', {}).get(cfg['filename_suffix'], {})
            keys = plot_override.get('keys', group['keys'])
            labels = plot_override.get('labels', group['labels'])
            data_list = []
            skip_plot = False

            for key in keys:
                if key not in results_store:
                    print(f"[SKIP] Group '{group['name']}' / plot '{cfg['filename_suffix']}': Missing key '{key}' in track-records.pkl")
                    skip_plot = True
                    break
                data_list.append(results_store[key])

            if skip_plot:
                continue

            filename = os.path.join(figures_dir, f"{group['name']}_{cfg['filename_suffix']}")

            plot_time_series_metric(
                data_list=data_list,
                labels=labels,
                param=cfg['param'],
                ylabel=cfg['ylabel'],
                filename=filename,
                legend_loc=cfg['legend_loc'],
                invert_y=cfg['invert_y'],
            )

    print("\nAll processing complete.")


if __name__ == '__main__':
    main()
