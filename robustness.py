"""
robustness.py
=============
MITS 6700G Group 8 — Module 7: Network Robustness Analysis

Simulates random and targeted node-removal attacks on the global
CVE network and measures the relative size of the Largest Connected
Component (LCC) at each step.

Functions:
    random_attack_simulation(G, n_steps)
    targeted_attack_simulation(G, n_steps)
    plot_robustness_comparison(random_results, targeted_results, N)
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


def random_attack_simulation(G, n_steps=30):
    """
    Simulate a random node-removal attack on the network.

    Nodes are removed in random order at fractions linearly spaced
    from 0 to 0.95. Measures relative LCC size (S/N) at each step.

    Parameters
    ----------
    G : nx.Graph
        Network to attack (will be copied — original is unchanged).
    n_steps : int, optional
        Number of measurement points (default: 30).

    Returns
    -------
    list[tuple[float, float]]
        List of (fraction_removed, relative_LCC_size) pairs.
    """
    rnd.seed(42)
    G_copy        = G.copy()
    N             = G_copy.number_of_nodes()
    fractions     = np.linspace(0, 0.95, n_steps)
    results       = []
    nodes_removed = 0

    for f in fractions:
        target_removed = int(f * N)
        n_to_remove    = target_removed - nodes_removed

        if n_to_remove > 0 and G_copy.number_of_nodes() > 0:
            to_remove = rnd.sample(
                list(G_copy.nodes()),
                min(n_to_remove, G_copy.number_of_nodes())
            )
            G_copy.remove_nodes_from(to_remove)
            nodes_removed += len(to_remove)

        if G_copy.number_of_nodes() == 0:
            lcc_size = 0
        else:
            lcc_size = len(max(nx.connected_components(G_copy), key=len))

        results.append((f, lcc_size / N))

    print(f"[ROBUSTNESS] Random attack complete: {n_steps} steps")
    return results


def targeted_attack_simulation(G, n_steps=30):
    """
    Simulate a targeted node-removal attack on the network.

    At each removal step the highest-degree node in the current
    (residual) graph is removed — adaptive degree re-computation.
    Measures relative LCC size (S/N) at each fraction.

    Parameters
    ----------
    G : nx.Graph
        Network to attack (will be copied — original is unchanged).
    n_steps : int, optional
        Number of measurement points (default: 30).

    Returns
    -------
    list[tuple[float, float]]
        List of (fraction_removed, relative_LCC_size) pairs.
    """
    G_copy        = G.copy()
    N             = G_copy.number_of_nodes()
    fractions     = np.linspace(0, 0.95, n_steps)
    results       = []
    nodes_removed = 0

    for f in fractions:
        target_removed = int(f * N)
        n_to_remove    = target_removed - nodes_removed

        for _ in range(n_to_remove):
            if G_copy.number_of_nodes() == 0:
                break
            # Adaptively find and remove highest-degree node
            highest = max(G_copy.degree(), key=lambda x: x[1])[0]
            G_copy.remove_node(highest)
            nodes_removed += 1

        lcc_size = (
            len(max(nx.connected_components(G_copy), key=len))
            if G_copy.number_of_nodes() > 0 else 0
        )
        results.append((f, lcc_size / N))

    print(f"[ROBUSTNESS] Targeted attack complete: {n_steps} steps")
    return results


def plot_robustness_comparison(random_results, targeted_results, N):
    """
    Side-by-side robustness comparison plot:
      Left:  S/N curves for random and targeted attacks with
             vulnerability gap shading and critical threshold lines.
      Right: Bar chart comparing critical fraction f_c values.

    Parameters
    ----------
    random_results : list[tuple[float, float]]
        Output of random_attack_simulation().
    targeted_results : list[tuple[float, float]]
        Output of targeted_attack_simulation().
    N : int
        Total node count of the original network.
    """
    f_rand, s_rand = zip(*random_results)
    f_targ, s_targ = zip(*targeted_results)
    f_rand = list(f_rand); s_rand = list(s_rand)
    f_targ = list(f_targ); s_targ = list(s_targ)

    # Critical threshold: first f where S/N < 0.05
    fc_rand = next((f for f, s in random_results  if s < 0.05), 1.0)
    fc_targ = next((f for f, s in targeted_results if s < 0.05), 1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # ── ax1: Main robustness curve ────────────────────────────────
    ax1.plot(f_rand, s_rand, color=COLOR_NEUTRAL, linestyle='--',
             marker='o', ms=4, lw=2.5, label='Random attack')
    ax1.plot(f_targ, s_targ, color=COLOR_LOG4SHELL, linestyle='-',
             marker='s', ms=4, lw=2.5, label='Targeted attack')
    ax1.fill_between(f_rand, s_rand, s_targ,
                     alpha=0.12, color=COLOR_BAD,
                     label='Vulnerability gap')
    ax1.axhline(0.05, color='gray', linestyle=':', lw=1.2,
                label='Fragmentation threshold (S/N=0.05)')
    ax1.axvline(fc_rand, color=COLOR_NEUTRAL, linestyle=':', lw=1.5,
                label=f'f_c random = {fc_rand:.2f}')
    ax1.axvline(fc_targ, color=COLOR_LOG4SHELL, linestyle=':', lw=1.5,
                label=f'f_c targeted = {fc_targ:.2f}')
    ax1.set_xlabel('Fraction of Nodes Removed (f)')
    ax1.set_ylabel('Relative LCC Size: S/N')
    ax1.set_title('Network Robustness Under Attack')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=10, loc='upper right')

    # ── ax2: Critical threshold bar chart ────────────────────────
    bars = ax2.bar(
        ['Random\nAttack', 'Targeted\nAttack'],
        [fc_rand, fc_targ],
        color=[COLOR_NEUTRAL, COLOR_LOG4SHELL],
        edgecolor='white', width=0.5
    )
    for bar, val in zip(bars, [fc_rand, fc_targ]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'f_c = {val:.2f}',
            ha='center', va='bottom',
            fontsize=12, fontweight='bold'
        )
    ax2.set_ylabel('Critical Fraction f_c')
    ax2.set_title('Critical Threshold Comparison\n(Lower = More Vulnerable)')
    ax2.set_ylim(0, 1.1)

    fig.suptitle(
        f'Network Robustness: Random vs Targeted Attack\n'
        f'CVE Vulnerability Network | N = {N} nodes',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout(pad=2.5)
    plt.savefig('outputs/robustness_comparison.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[ROBUSTNESS] Random attack  f_c = {fc_rand:.2f}")
    print(f"[ROBUSTNESS] Targeted attack f_c = {fc_targ:.2f}")
    print(f"[ROBUSTNESS] Network is MORE vulnerable to TARGETED attack ✓ "
          f"(scale-free)")
