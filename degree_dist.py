"""
degree_dist.py
==============
MITS 6700G Group 8 — Module 3: Degree Distribution Analysis

Produces publication-quality side-by-side linear + log-log degree
distribution plots with power-law fit for both graph layers.

Functions:
    plot_degree_distribution(G, graph_name, title_suffix)
"""

import numpy as np
import networkx as nx

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


def plot_degree_distribution(G, graph_name="graph", title_suffix=""):
    """
    Plot side-by-side linear + log-log degree distribution with
    power-law fit for a given graph.

    Saves figure to outputs/degree_dist_{graph_name}.png.

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph to analyse.
    graph_name : str, optional
        Short identifier used in the filename (e.g., 'global', 'local').
    title_suffix : str, optional
        Additional subtitle text (e.g., layer label).

    Returns
    -------
    float
        Fitted power-law exponent gamma (α).
    """
    import powerlaw  # imported here to avoid top-level failure if not installed

    degrees  = [d for _, d in G.degree()]
    N        = G.number_of_nodes()
    unique_k, counts = np.unique(degrees, return_counts=True)
    pk       = counts / counts.sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ── ax1: LINEAR SCALE ─────────────────────────────────────────
    ax1.scatter(unique_k, pk, color=COLOR_NEUTRAL, alpha=0.7, s=30, zorder=3)
    ax1.axvline(
        np.mean(degrees),
        color=COLOR_HIGHLIGHT, linestyle='--', linewidth=1.8,
        label=f'Mean = {np.mean(degrees):.2f}'
    )
    ax1.set_xlabel('Degree k')
    ax1.set_ylabel('Probability P(k)')
    ax1.set_title('Degree Distribution (Linear Scale)')
    ax1.legend()

    # ── ax2: LOG-LOG SCALE ────────────────────────────────────────
    ax2.scatter(unique_k, pk, color=COLOR_NEUTRAL, alpha=0.7, s=30, zorder=3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Degree k (log scale)')
    ax2.set_ylabel('Probability P(k) (log scale)')
    ax2.set_title('Degree Distribution (Log-Log Scale)')
    # Minor gridlines for log-log plot
    ax2.grid(True, which='minor', alpha=0.15)

    # ── Power-law fit ─────────────────────────────────────────────
    fit    = powerlaw.Fit(degrees, discrete=True, verbose=False)
    gamma  = fit.power_law.alpha
    xmin   = fit.power_law.xmin

    x_range = np.logspace(
        np.log10(max(xmin, unique_k.min())),
        np.log10(max(degrees)),
        100
    )
    y_fit   = (x_range / xmin) ** (-gamma)
    y_fit  /= y_fit.sum()

    ax2.plot(x_range, y_fit, color=COLOR_LOG4SHELL, linewidth=2.5,
             label=f'Power-law fit: γ = {gamma:.3f}')
    ax2.legend()

    # ── Text annotation box ───────────────────────────────────────
    scale_free_label = "YES ✓" if 2.0 < gamma < 3.5 else "CHECK"
    textstr = (f'N = {N} nodes\n'
               f'γ = {gamma:.3f}\n'
               f'x_min = {xmin:.1f}\n'
               f'Scale-free: {scale_free_label}')
    ax2.text(
        0.97, 0.97, textstr, transform=ax2.transAxes,
        fontsize=10, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                  edgecolor='gray', alpha=0.9)
    )

    fig.suptitle(
        f'Degree Distribution — {graph_name}\n{title_suffix}',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout(pad=2.5)
    plt.savefig(
        f'outputs/degree_dist_{graph_name}.png',
        dpi=300, bbox_inches='tight', facecolor='white'
    )
    plt.close()

    print(f"[DEGREE_DIST] {graph_name}: N={N}, γ={gamma:.4f}")
    return gamma
