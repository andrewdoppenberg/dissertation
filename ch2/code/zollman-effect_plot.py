import os
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ── Font / style ──────────────────────────────────────────────────────────────
size = 24
mpl.rcParams['font.family'] = 'Garamond'
mpl.rcParams['font.size'] = size
mpl.rcParams['axes.titlesize'] = size
mpl.rcParams['axes.labelsize'] = size
mpl.rcParams['xtick.labelsize'] = size
mpl.rcParams['ytick.labelsize'] = size
mpl.rcParams['legend.fontsize'] = size*0.9

# ── Load results ──────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
default_path = os.path.join(script_dir, "..", "results", "zollman-effect.pkl")
data_path = sys.argv[1] if len(sys.argv) > 1 else default_path

print(f"Loading results from '{data_path}'...")
with open(data_path, "rb") as f:
    output = pickle.load(f)

data            = output['data']
TIME_STEPS      = output['TIME_STEPS']
MAX_STEPS       = output['MAX_STEPS']
DELTAS          = output['DELTAS']
TRIALS          = output['TRIALS']
MISTRUST_LEVELS = output['MISTRUST_LEVELS']
DELAYS          = output.get('DELAYS', [0])


figures_dir = os.path.join(script_dir, "..", "figures")

os.makedirs(figures_dir, exist_ok=True)

metrics = ['consensus', 'credence', 'mse', 'mse_non_consensus', 'mse_consensus', 'min_evidence']

# ── 1. Heatmaps for all metrics ───────────────────────────────────────────────
print("Generating heatmaps...")
for metric in metrics:
    for n in TRIALS:
        heatmap_data = np.zeros((len(DELTAS), len(TIME_STEPS)))

        for i, delta in enumerate(DELTAS):
            for j, t in enumerate(TIME_STEPS):
                val_full_trust = np.array(data[delta][n][MISTRUST_LEVELS[-1]][t][0][metric][0])
                val_low_trust  = np.array(data[delta][n][MISTRUST_LEVELS[0]][t][0][metric][0])

                if metric in ['mse', 'mse_non_consensus', 'mse_consensus']:
                    diff = val_low_trust - val_full_trust
                else:
                    diff = val_full_trust - val_low_trust

                heatmap_data[i, j] = diff

        plt.figure(figsize=(8, 6))

        if metric in ['mse', 'mse_non_consensus', 'mse_consensus']:
            fmt_str = ".3f"
        elif metric == 'min_evidence':
            fmt_str = ".0f"
        else:
            fmt_str = ".2f"

        ax = sns.heatmap(
            heatmap_data,
            xticklabels=TIME_STEPS,
            yticklabels=DELTAS,
            cmap="RdBu_r",
            center=0,
            annot=True,
            fmt=fmt_str,
            annot_kws={"weight": "bold"},
            cbar_kws={'label': 'Difference (Red = Full Trust Better)'},
        )

        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(2)

        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Utility Difference")

        plt.tight_layout()
        out_path = os.path.join(figures_dir, f"{metric}_trials_{n}.pdf")
        plt.savefig(out_path, format="pdf")
        plt.close()
        print(f"  Saved {out_path}")

# ── 2. Time plots for consensus ───────────────────────────────────────────────
print("Generating time plots...")
for metric in ['consensus','credence','mse']:
    plt.figure(figsize=(10, 6))

    plot_steps    = 10_000
    all_steps_arr = np.arange(0, plot_steps + 1)

    target_delta  = 0.001
    target_trials = 1000

    for w in MISTRUST_LEVELS:
        avg_over_time = np.array(
            data[target_delta][target_trials][w][MAX_STEPS][0]['consensus_all_steps']
        )[:plot_steps + 1]
        lbl = f'w = {w}'
        plt.plot(all_steps_arr, avg_over_time, linewidth=2, label=lbl)

    plt.ylim(0.4, 0.75)
    plt.xlabel("Time Steps")
    plt.ylabel("True Consensus Rate (TCR)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(figures_dir, f"{metric}_over_time.pdf")
    plt.savefig(out_path, format="pdf")
    plt.close()
    print(f"  Saved {out_path}")

# ── 2b. Time plots for divergence ─────────────────────────────────────────────
print("Generating divergence time plots...")
target_delta = 0.001
for t_trials in [1, 10]:
    plt.figure(figsize=(10, 6))

    plot_steps    = 10_000
    div_steps_arr = np.arange(1, plot_steps + 1)

    for w in MISTRUST_LEVELS:
        try:
            div_over_time = np.array(
                data[target_delta][t_trials][w][MAX_STEPS][0]['divergence_trajectory']
            )[:plot_steps]

            if w == MISTRUST_LEVELS[0]:
                lbl = f'Low Trust ({w})'
            elif w == MISTRUST_LEVELS[-1]:
                lbl = f'Full Trust ({w})'
            else:
                lbl = f'Moderate Distrust ({w})'

            plt.plot(div_steps_arr[:len(div_over_time)], div_over_time, linewidth=2, label=lbl)
        except KeyError:
            print(f"  Missing divergence data for trials={t_trials}, w={w}")

    plt.xlabel("Time Steps")
    plt.ylabel("Mean Absolute Divergence in Choice")
    plt.title(f"Between-Group Choice Divergence (Trials={t_trials})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')  # Log scale makes early transients visible over 10,000 steps
    plt.tight_layout()

    out_path = os.path.join(figures_dir, f"divergence_trials_{t_trials}.pdf")
    plt.savefig(out_path, format="pdf")
    plt.close()
    print(f"  Saved {out_path}")

print("Generating MSE histograms...")
# ── 3. Histograms for MSE per run ──────────────────────────────────────────────
# Compare w=1.0 and w=0.01 at specific simulation parameters.
target_time = 1000
target_delta = 0.001
target_trials = 10

w_full = 1.0
w_low = 0.01

x0 = 0
x1 = 0.2

try:
    mse_full = np.array(data[target_delta][target_trials][w_full][target_time][0]['mse_non_consensus_per_run'])
    mse_low = np.array(data[target_delta][target_trials][w_low][target_time][0]['mse_non_consensus_per_run'])

    plt.figure(figsize=(10, 6))
    bins = np.linspace(x0, x1, 50)
    
    plt.hist(mse_full, bins=bins, color='blue', edgecolor='black', label=f'Full Trust ({w_full})', density=True, alpha=0.5, log=True)
    plt.hist(mse_low, bins=bins, color='red', edgecolor='black', label=f'Low Trust ({w_low})', density=True, alpha=0.5, log=True)
    
    try:
        frac_full = data[target_delta][target_trials][w_full][target_time][0]['fraction_agents_less_than_1'][0] * 100
        frac_low = data[target_delta][target_trials][w_low][target_time][0]['fraction_agents_less_than_1'][0] * 100
        textstr = (f"Groups w/ <1 effective evidence about a theory:\n"
                   f"Full Trust: {frac_full:.2f}%\n"
                   f"Low Trust: {frac_low:.2f}%")
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black')
        plt.gca().text(0.95, 0.5, textstr, transform=plt.gca().transAxes, fontsize=12,
                       verticalalignment='center', horizontalalignment='right', bbox=props)
    except KeyError:
        pass

    plt.xlabel("Utility Accuracy (UA) - Non-Consensus Theory")
    plt.ylabel("Number of Runs")
    plt.xlim(x0, x1)
    #plt.title(f"MSE of Non-Consensus Theory Distribution (T={target_time}, δ={target_delta}, Trials={target_trials})")
    plt.legend()
    plt.tight_layout()
    
    out_path = os.path.join(figures_dir, "mse_histogram.pdf")
    plt.savefig(out_path, format="pdf")
    plt.close()
    print(f"  Saved {out_path}")
except KeyError:
    print("  'mse_non_consensus_per_run' not found in data. Skipping MSE histograms.")

print("Generating Max MSE histograms...")
# ── 4. Histograms for maximum MSE per run ─────────────────────────────────────
try:
    max_mse_full = np.array(data[target_delta][target_trials][w_full][target_time][0]['max_mse_per_run'])
    max_mse_low = np.array(data[target_delta][target_trials][w_low][target_time][0]['max_mse_per_run'])

    plt.figure(figsize=(10, 6))
    
    # We use a reasonable max bin based on the data. For max MSE, it could realistically approach 1.0 (though rarely reaches it)
    max_val = max(np.max(max_mse_full), np.max(max_mse_low))
    bins = np.linspace(0, max_val, 50)
    
    plt.hist(max_mse_full, bins=bins, color='blue', edgecolor='black', label=f'Full Trust ({w_full})', density=True, alpha=0.5, log=True)
    plt.hist(max_mse_low, bins=bins, color='red', edgecolor='black', label=f'Low Trust ({w_low})', density=True, alpha=0.5, log=True)
    
    try:
        frac_full = data[target_delta][target_trials][w_full][target_time][0]['fraction_agents_less_than_1'][0] * 100
        frac_low = data[target_delta][target_trials][w_low][target_time][0]['fraction_agents_less_than_1'][0] * 100
        textstr = (f"Groups w/ <1 effective evidence about a theory:\n"
                   f"Full Trust: {frac_full:.2f}%\n"
                   f"Low Trust: {frac_low:.2f}%")
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black')
        plt.gca().text(0.95, 0.5, textstr, transform=plt.gca().transAxes, fontsize=12,
                       verticalalignment='center', horizontalalignment='right', bbox=props)
    except KeyError:
        pass

    plt.xlabel("Worst Mean Squared Error (MSE) in a Run")
    plt.ylabel("Density")
    plt.xlim(0, max_val)
    #plt.title(f"Max MSE Distribution (T={target_time}, δ={target_delta}, Trials={target_trials})")
    plt.legend()
    plt.tight_layout()
    
    out_path = os.path.join(figures_dir, "max_mse_histogram.pdf")
    plt.savefig(out_path, format="pdf")
    plt.close()
    print(f"  Saved {out_path}")
except KeyError:
    print("  'max_mse_per_run' not found in data. Skipping Max MSE histograms.")

print("Generating Min Evidence histograms...")
# ── 5. Histograms for minimum effective evidence per run ──────────────────────
for t_trials in [1, 10]:
    try:
        min_ev_full = np.array(data[target_delta][t_trials][w_full][target_time][0]['min_evidence_per_run'])
        min_ev_low = np.array(data[target_delta][t_trials][w_low][target_time][0]['min_evidence_per_run'])
        
        mse_full_run = np.array(data[target_delta][t_trials][w_full][target_time][0]['mse_per_run'])
        mse_low_run = np.array(data[target_delta][t_trials][w_low][target_time][0]['mse_per_run'])
    
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        max_val = max(np.max(min_ev_full), np.max(min_ev_low))
        max_val = 400
        bins = np.linspace(0, max_val, 41)
        
        ax1.hist(min_ev_full, bins=bins, color='blue', edgecolor='black', label=f'Full Trust (w={w_full})', density=True, alpha=0.4, log=True)
        ax1.hist(min_ev_low, bins=bins, color='red', edgecolor='black', label=f'Low Trust (w={w_low})', density=True, alpha=0.4, log=True)
        
        ax1.set_xlabel("Minimum Effective Evidence a Group Receives about a Theory")
        ax1.set_ylabel("Probability of Result")
        ax1.set_xlim(0, max_val)
        
        # Calculate average MSE per bin
        bin_centers = (bins[:-1] + bins[1:]) / 2
        avg_mse_full = np.zeros_like(bin_centers)
        avg_mse_low = np.zeros_like(bin_centers)
        
        for i in range(len(bins)-1):
            mask_f = (min_ev_full >= bins[i]) & (min_ev_full < bins[i+1])
            avg_mse_full[i] = np.mean(mse_full_run[mask_f]) if np.any(mask_f) else np.nan
            
            mask_l = (min_ev_low >= bins[i]) & (min_ev_low < bins[i+1])
            avg_mse_low[i] = np.mean(mse_low_run[mask_l]) if np.any(mask_l) else np.nan
    
        ax2 = ax1.twinx()
        ax2.plot(bin_centers, avg_mse_full, color='darkblue', marker='o', linestyle='-', linewidth=2, label=f'Full Trust MSE')
        ax2.plot(bin_centers, avg_mse_low, color='darkred', marker='x', linestyle='-', linewidth=2, label=f'Low Trust MSE')
        ax2.set_ylabel("MSE of Entire Network")
    
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')
    
        plt.tight_layout()
        
        out_path = os.path.join(figures_dir, f"min_evidence_histogram_{t_trials}.pdf")
        plt.savefig(out_path, format="pdf")
        plt.close()
        print(f"  Saved {out_path}")
    except KeyError:
        print(f"  'min_evidence_per_run' not found in data for trials={t_trials}. Skipping.")

# ── 6. Delayed distrust plot ──────────────────────────────────────────────────
print("Generating delayed distrust plots...")
target_delta = 0.001
target_trials = 1000
target_steps = 10000

for d in DELAYS:
    if d == 0:
        continue # Skip normal sweep

    plt.figure(figsize=(10, 6))
    
    for w in MISTRUST_LEVELS:
        try:
            cons_delayed = data[target_delta][target_trials][w][target_steps][d]['consensus_all_steps']
            lbl = f'w = {w} (at t = {d})' if w < 1.0 else 'w=1.0'
            plt.plot(cons_delayed, label=lbl, linewidth=2)
        except KeyError:
            print(f"  Missing delayed distrust data for w={w}, d={d}")

    plt.ylim(0.45, 0.7)
    plt.xlabel('Time Steps')
    plt.ylabel('True Consensus Rate (TCR)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_path = os.path.join(figures_dir, f"delayed_distrust_{d}.pdf")
    plt.savefig(out_path, format="pdf")
    plt.close()
    print(f"  Saved {out_path}")

print(f"Done. Visualizations saved in '{figures_dir}'.")