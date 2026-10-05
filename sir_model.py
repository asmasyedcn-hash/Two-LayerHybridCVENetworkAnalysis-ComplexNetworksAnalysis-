"""
sir_model.py
============
MITS 6700G Group 8 — Module 6: SIR Epidemic Model

Discrete-time SIR simulation seeded from Log4Shell (CVE-2021-44228).
Results are averaged across n_runs for stability (seed 42+run).

Fixes applied:
  FIX-2: N defined as first line of sir_simulation(); added to return dict
  FIX-3: plot_sir_single() uses result['N'] (not result['s_curve'][0])
  FIX-7: compare_beta_values() uses result['N'] for fraction infected

Functions:
    sir_simulation(G, seed_node, beta, gamma_rate, max_steps, n_runs)
    plot_sir_single(result, beta, gamma_rate, graph_label)
    compare_beta_values(G, seed_node, beta_values, gamma_rate)
    hop_reachability_plot(G, seed_node, max_hops)
"""

import numpy as np
import networkx as nx
import random as rnd

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — must be FIRST
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Global style ────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.figsize':       (12, 8),
    'figure.dpi':           150,
    'figure.facecolor':     'white',
    'figure.edgecolor':     'white',
    'font.family':          'DejaVu Sans',
    'font.size':            11,
    'axes.titlesize':       16,
    'axes.titleweight':     'bold',
    'axes.titlepad':        20,
    'axes.labelsize':       13,
    'axes.labelweight':     'bold',
    'axes.labelpad':        10,
    'axes.facecolor':       '#FAFAFA',
    'axes.edgecolor':       '#CCCCCC',
    'axes.linewidth':       1.2,
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'axes.grid':            True,
    'grid.alpha':           0.3,
    'grid.linestyle':       '--',
    'grid.color':           'gray',
    'xtick.labelsize':      11,
    'ytick.labelsize':      11,
    'xtick.direction':      'out',
    'ytick.direction':      'out',
    'legend.fontsize':      11,
    'legend.framealpha':    0.9,
    'legend.edgecolor':     'gray',
    'legend.fancybox':      True,
    'lines.linewidth':      2.0,
    'lines.markersize':     6,
    'savefig.dpi':          300,
    'savefig.bbox':         'tight',
    'savefig.facecolor':    'white',
    'savefig.pad_inches':   0.2,
})

# ── Color constants ─────────────────────────────────────────────────
PALETTE_MAIN    = sns.color_palette("husl", 10)
PALETTE_COMM    = sns.color_palette("tab20", 20)
COLOR_LOG4SHELL = '#E63946'
COLOR_HIGHLIGHT = '#F4A261'
COLOR_NEUTRAL   = '#457B9D'
COLOR_GOOD      = '#2A9D8F'
COLOR_BAD       = '#E76F51'


def sir_simulation(G, seed_node, beta, gamma_rate,
                   max_steps=100, n_runs=10):
    """
    Discrete-time SIR epidemic simulation seeded from a given node.

    Averages S, I, R curves across n_runs for numerical stability.
    Each run uses seed (42 + run) for reproducibility with variation.

    FIX-2 APPLIED: N is defined first; N is included in return dict.

    Parameters
    ----------
    G : nx.Graph
        Network to simulate on.
    seed_node : str
        Starting infected node (e.g., 'CVE-2021-44228').
    beta : float
        Transmission probability per susceptible neighbour per step.
    gamma_rate : float
        Recovery probability per infected node per step.
    max_steps : int, optional
        Maximum simulation steps (default: 100).
    n_runs : int, optional
        Number of independent runs to average (default: 10).

    Returns
    -------
    dict or None
        Keys: s_curve, i_curve, r_curve, peak_infected, peak_time,
              total_infected, n_steps, N.
        Returns None if seed_node not in G.
    """
    # FIX-2: Define N FIRST
    N = G.number_of_nodes()

    if seed_node not in G:
        print(f"[SIR] ✗ Seed node {seed_node} not in graph — skipping")
        return None

    all_s, all_i, all_r = [], [], []

    for run in range(n_runs):
        rnd.seed(42 + run)  # reproducible but varied across runs
        S = set(G.nodes()) - {seed_node}
        I = {seed_node}
        R = set()

        s_curve = [len(S)]
        i_curve = [len(I)]
        r_curve = [len(R)]

        for _ in range(max_steps):
            new_I = set()
            new_R = set()

            for node in I:
                for nbr in G.neighbors(node):
                    if nbr in S and rnd.random() < beta:
                        new_I.add(nbr)
                if rnd.random() < gamma_rate:
                    new_R.add(node)

            S -= new_I
            I  = (I | new_I) - new_R
            R |= new_R

            s_curve.append(len(S))
            i_curve.append(len(I))
            r_curve.append(len(R))

            if len(I) == 0:
                break

        all_s.append(s_curve)
        all_i.append(i_curve)
        all_r.append(r_curve)

    # Pad all curves to the same length (repeat last value)
    max_len = max(len(c) for c in all_i)

    def pad(curves):
        return [c + [c[-1]] * (max_len - len(c)) for c in curves]

    s_avg = list(np.mean(pad(all_s), axis=0))
    i_avg = list(np.mean(pad(all_i), axis=0))
    r_avg = list(np.mean(pad(all_r), axis=0))

    peak_val = max(i_avg)
    t_peak   = i_avg.index(peak_val)
    r_final  = r_avg[-1]

    print(f"[SIR] beta={beta}, gamma={gamma_rate}, seed={seed_node}")
    print(f"[SIR] N (total nodes)       : {N}")
    print(f"[SIR] Peak infected         : {peak_val:.1f} "
          f"({peak_val/N*100:.1f}% of N)")
    print(f"[SIR] Peak at timestep      : {t_peak}")
    print(f"[SIR] Total infected (R end): {r_final:.1f} "
          f"({r_final/N*100:.1f}% of N)")

    return {
        's_curve':        s_avg,
        'i_curve':        i_avg,
        'r_curve':        r_avg,
        'peak_infected':  peak_val,
        'peak_time':      t_peak,
        'total_infected': r_final,
        'n_steps':        max_len,
        'N':              N,       # FIX-2: N in return dict
    }


def plot_sir_single(result, beta, gamma_rate, graph_label):
    """
    Plot a single SIR simulation result: S(t), I(t), R(t) curves
    with filled areas, peak annotation and summary text box.

    FIX-3 APPLIED: uses result['N'] everywhere.

    Parameters
    ----------
    result : dict
        Return value from sir_simulation().
    beta : float
        Transmission probability (used in title and filename).
    gamma_rate : float
        Recovery probability (used in title).
    graph_label : str
        Human-readable layer description for the subtitle.
    """
    # FIX-3: Use result['N']
    N      = result['N']
    t_axis = list(range(result['n_steps']))

    fig, ax = plt.subplots(figsize=(13, 7))

    ax.plot(t_axis, result['s_curve'], color='#2196F3', lw=2.5,
            label='S(t) — Susceptible')
    ax.plot(t_axis, result['i_curve'], color=COLOR_LOG4SHELL, lw=2.5,
            label='I(t) — Infected')
    ax.plot(t_axis, result['r_curve'], color=COLOR_GOOD, lw=2.5,
            label='R(t) — Recovered')

    ax.fill_between(t_axis, result['s_curve'], alpha=0.12, color='#2196F3')
    ax.fill_between(t_axis, result['i_curve'], alpha=0.12,
                    color=COLOR_LOG4SHELL)
    ax.fill_between(t_axis, result['r_curve'], alpha=0.12, color=COLOR_GOOD)

    # Peak marker
    tp = result['peak_time']
    pk = result['peak_infected']
    ax.axvline(tp, color='gray', linestyle='--', lw=1.5, alpha=0.7)
    ax.annotate(
        f'Peak: {pk:.0f} nodes ({pk/N*100:.1f}%)',
        xy=(tp, pk),
        xytext=(tp + max(t_axis) * 0.06, pk * 0.85),
        fontsize=10, color=COLOR_LOG4SHELL,
        arrowprops=dict(arrowstyle='->', color=COLOR_LOG4SHELL)
    )

    # FIX-3: Summary box uses result['N']
    summary = (f'N (total nodes) = {N}\n'
               f'β = {beta} | γ = {gamma_rate}\n'
               f'Peak infected   : {pk:.0f} ({pk/N*100:.1f}%)\n'
               f'Total infected  : {result["total_infected"]:.0f} '
               f'({result["total_infected"]/N*100:.1f}%)')
    ax.text(0.02, 0.97, summary, transform=ax.transAxes,
            fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.9))

    ax.set_xlabel('Time Step t')
    ax.set_ylabel('Number of Nodes')
    ax.set_title(
        f'SIR Epidemic Model — Log4Shell Seed (CVE-2021-44228)\n'
        f'β={beta}, γ={gamma_rate} | {graph_label}',
        fontweight='bold'
    )
    ax.legend(loc='center right', framealpha=0.9)

    plt.tight_layout(pad=2.0)
    plt.savefig(
        f'outputs/sir_single_beta{beta}.png',
        dpi=300, bbox_inches='tight', facecolor='white'
    )
    plt.close()
    print(f"[SIR] Saved: outputs/sir_single_beta{beta}.png")


def compare_beta_values(G, seed_node, beta_values, gamma_rate=0.1):
    """
    Run SIR simulations for multiple β values and produce a
    side-by-side comparison: I(t) curves and total fraction infected.

    FIX-7 APPLIED: fraction infected = total_infected / result['N'].

    Parameters
    ----------
    G : nx.Graph
        Network to simulate on.
    seed_node : str
        Starting infected node.
    beta_values : list[float]
        List of transmission probabilities to compare.
    gamma_rate : float, optional
        Recovery probability (default: 0.1).
    """
    # FIX-7: define N from graph
    N = G.number_of_nodes()

    results = {}
    for beta in beta_values:
        results[beta] = sir_simulation(
            G, seed_node, beta, gamma_rate,
            max_steps=100, n_runs=20
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(beta_values)))

    # ax1 — I(t) curves
    for (beta, res), color in zip(results.items(), colors):
        if res is not None:
            ax1.plot(res['i_curve'], color=color, lw=2.5,
                     label=f'β = {beta}')
    ax1.set_xlabel('Time Step t')
    ax1.set_ylabel('Infected Nodes I(t)')
    ax1.set_title('Infected I(t) — Comparing β Values')
    ax1.legend(fontsize=11)

    # ax2 — Bar chart: total fraction infected
    # FIX-7: fraction = total_infected / result['N']
    valid_betas     = [b for b, r in results.items() if r is not None]
    fractions       = [results[b]['total_infected'] / results[b]['N']
                       for b in valid_betas]
    bar_colors      = plt.cm.Reds(
        np.linspace(0.3, 0.9, len(valid_betas))
    )

    bars = ax2.bar(
        [str(b) for b in valid_betas], fractions,
        color=bar_colors, edgecolor='white', linewidth=0.8
    )
    for bar, frac in zip(bars, fractions):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{frac*100:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
    ax2.set_xlabel('Transmission Probability β')
    ax2.set_ylabel('Fraction of Network Infected')
    ax2.set_title('Total Fraction Infected by β Value')
    ax2.set_ylim(0, 1.05)

    fig.suptitle(
        'SIR Epidemic Sensitivity Analysis\n'
        'Log4Shell as Seed Node | Varying Transmission Rate β',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout(pad=2.5)
    plt.savefig('outputs/sir_beta_comparison.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[SIR] Saved: outputs/sir_beta_comparison.png")


def hop_reachability_plot(G, seed_node='CVE-2021-44228', max_hops=4):
    """
    Plot exact and cumulative CVE reachability from Log4Shell
    at each hop distance (1 to max_hops).

    Left panel:  exact count per hop distance.
    Right panel: cumulative count and percentage of full network.

    Parameters
    ----------
    G : nx.Graph
        Network to analyse (Layer 1 global graph).
    seed_node : str, optional
        Source node (default: 'CVE-2021-44228').
    max_hops : int, optional
        Maximum hop distance to analyse (default: 4).
    """
    N = G.number_of_nodes()

    hop_list   = []
    exact_list = []
    cumul_list = []
    pct_list   = []

    for k in range(1, max_hops + 1):
        path_lengths = nx.single_source_shortest_path_length(
            G, seed_node, cutoff=k
        )
        exact_k = sum(1 for v in path_lengths.values() if v == k)
        cumul_k = sum(1 for v in path_lengths.values() if v <= k)
        hop_list.append(k)
        exact_list.append(exact_k)
        cumul_list.append(cumul_k)
        pct_list.append(cumul_k / N * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ax1 — Exact reachable per hop
    bars1 = ax1.bar(hop_list, exact_list, color=COLOR_NEUTRAL,
                    edgecolor='white', width=0.6)
    for bar, cnt, k in zip(bars1, exact_list, hop_list):
        pct_exact = cnt / N * 100
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(exact_list) * 0.01,
            f'{cnt}\n({pct_exact:.1f}%)',
            ha='center', va='bottom', fontsize=9, fontweight='bold'
        )
    ax1.set_title('CVEs Reachable at Exactly k Hops from Log4Shell')
    ax1.set_xlabel('Hop Distance k')
    ax1.set_ylabel('Number of CVEs')
    ax1.set_xticks(hop_list)

    # ax2 — Cumulative
    bars2 = ax2.bar(hop_list, cumul_list, color=COLOR_LOG4SHELL,
                    alpha=0.8, edgecolor='white', width=0.6,
                    label='Cumulative count')
    for bar, cnt in zip(bars2, cumul_list):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(cumul_list) * 0.01,
            f'{cnt}', ha='center', va='bottom',
            fontsize=9, fontweight='bold'
        )

    twin = ax2.twinx()
    twin.plot(hop_list, pct_list, 'k--', marker='o', lw=2,
              label='% of network')
    twin.set_ylabel('% of Total Network', fontsize=11)
    # Make twin axis invisible to keep style clean
    twin.tick_params(axis='y', labelsize=10)

    ax2.set_title('Cumulative CVEs Reachable Within k Hops')
    ax2.set_xlabel('Hop Distance k')
    ax2.set_ylabel('Cumulative CVE Count')
    ax2.set_xticks(hop_list)
    ax2.legend(loc='upper left', fontsize=10)

    fig.suptitle(
        f'Structural Reachability from Log4Shell (CVE-2021-44228)\n'
        f'Total network: N = {N} nodes',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout(pad=2.5)
    plt.savefig('outputs/hop_reachability.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[SIR] Saved: outputs/hop_reachability.png")
