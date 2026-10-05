"""
summary_dashboard.py
====================
MITS 6700G Group 8 — Module 8: Summary Dashboard

One-page 2×3 summary figure combining all analysis results into a
single publication-quality report cover figure.

Panels:
  [0,0] CVSS score distribution histogram
  [0,1] Top 10 CVEs by PageRank (compact)
  [0,2] Community sizes (compact)
  [1,0] Degree distribution log-log (compact)
  [1,1] SIR I(t) curve (β=0.3)
  [1,2] Robustness: Random vs Targeted (compact)

Functions:
    plot_summary_dashboard(df_global_cent, df_raw, partition,
                           modularity, sir_result,
                           n_communities, gamma_exp,
                           random_results, targeted_results)
"""

import numpy as np
import pandas as pd
from collections import Counter

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


def plot_summary_dashboard(df_global_cent, df_raw, partition,
                            modularity, sir_result,
                            n_communities, gamma_exp,
                            random_results, targeted_results):
    """
    Produce a one-page 2×3 summary dashboard figure combining all
    key analysis results for the project report.

    Parameters
    ----------
    df_global_cent : pd.DataFrame
        Global layer centrality DataFrame (with cve_id, pagerank,
        cvss_score, degree_cent columns).
    df_raw : pd.DataFrame
        Full parsed CVE DataFrame (for CVSS histogram).
    partition : dict
        Node→community mapping from Louvain.
    modularity : float
        Louvain modularity Q score.
    sir_result : dict
        Return value from sir_simulation() at β=0.3.
    n_communities : int
        Total number of communities detected.
    gamma_exp : float
        Power-law exponent γ from degree distribution fit.
    random_results : list[tuple[float, float]]
        Output of random_attack_simulation().
    targeted_results : list[tuple[float, float]]
        Output of targeted_attack_simulation().
    """
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.suptitle(
        'CVE Vulnerability Network — Analysis Summary\n'
        'Anchored on Log4Shell (CVE-2021-44228) | MITS 6700G Group 8',
        fontsize=16, fontweight='bold', y=1.01
    )
    fig.patch.set_facecolor('white')

    # ── PANEL [0,0]: CVSS Score Distribution ─────────────────────
    ax = axes[0, 0]
    cvss = df_raw['cvss_score'].dropna()
    ax.hist(cvss, bins=30, color=COLOR_NEUTRAL, alpha=0.7,
            edgecolor='white')
    ax.axvline(7.0, color=COLOR_BAD, ls='--', lw=1.5,
               label='CVSS 7.0 (High threshold)')
    ax.axvline(10.0, color=COLOR_LOG4SHELL, ls='-', lw=2.0,
               label='Log4Shell CVSS=10.0')
    ax.set_xlabel('CVSS Score', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.set_title('CVSS Score Distribution', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)

    # ── PANEL [0,1]: Top 10 PageRank (compact) ───────────────────
    ax = axes[0, 1]
    top10_pr = df_global_cent.nlargest(10, 'pagerank')
    colors = [
        COLOR_LOG4SHELL if 'CVE-2021-44228' in str(r.cve_id)
        else COLOR_NEUTRAL
        for _, r in top10_pr.iterrows()
    ]
    ax.barh(range(10), top10_pr['pagerank'], color=colors,
            height=0.7, edgecolor='white')
    ax.invert_yaxis()
    ax.set_yticks(range(10))
    ax.set_yticklabels(
        [r.cve_id[-13:] for _, r in top10_pr.iterrows()],
        fontsize=8
    )
    ax.set_xlabel('PageRank', fontsize=9)
    ax.set_title('Top 10 CVEs by PageRank', fontsize=11,
                 fontweight='bold')

    # ── PANEL [0,2]: Community Sizes (compact) ───────────────────
    ax = axes[0, 2]
    N_global = len(df_global_cent)
    sizes    = Counter(partition.values())
    log4_c   = partition.get('CVE-2021-44228', -1)
    top15    = sorted(sizes.items(), key=lambda x: -x[1])[:15]
    bar_colors = [
        COLOR_LOG4SHELL if c == log4_c else COLOR_NEUTRAL
        for c, _ in top15
    ]
    ax.bar(range(len(top15)), [s for _, s in top15],
           color=bar_colors, edgecolor='white')
    ax.set_xlabel('Community (rank)', fontsize=9)
    ax.set_ylabel('Size', fontsize=9)
    ax.set_title(f'Community Sizes | Q={modularity:.3f}',
                 fontsize=11, fontweight='bold')

    # ── PANEL [1,0]: Degree Distribution log-log (compact) ───────
    ax = axes[1, 0]
    # Reconstruct degrees from degree_cent: degree = degree_cent * (N-1)
    N_g = N_global
    raw_degrees = (df_global_cent['degree_cent'] * (N_g - 1)).values
    raw_degrees = raw_degrees[raw_degrees > 0].astype(int)

    if len(raw_degrees) > 0:
        unique_k, counts = np.unique(raw_degrees, return_counts=True)
        pk = counts / counts.sum()
        ax.scatter(unique_k, pk, color=COLOR_NEUTRAL, alpha=0.6,
                   s=20, zorder=3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        # Simple power-law line overlay using gamma_exp
        if len(unique_k) > 1:
            x_range = np.logspace(
                np.log10(unique_k.min()),
                np.log10(unique_k.max()),
                50
            )
            x_min_val = unique_k.min()
            y_fit = (x_range / x_min_val) ** (-gamma_exp)
            y_fit /= y_fit.sum()
            ax.plot(x_range, y_fit, color=COLOR_LOG4SHELL, lw=1.8)
    ax.set_xlabel('Degree k', fontsize=9)
    ax.set_ylabel('P(k)', fontsize=9)
    ax.set_title(f'Degree Distribution | γ={gamma_exp:.3f}',
                 fontsize=11, fontweight='bold')
    ax.grid(True, which='minor', alpha=0.1)

    # ── PANEL [1,1]: SIR I(t) Curve ──────────────────────────────
    ax = axes[1, 1]
    N_sir  = sir_result['N']
    t_axis = range(sir_result['n_steps'])
    ax.plot(t_axis, sir_result['i_curve'],
            color=COLOR_LOG4SHELL, lw=2.5)
    ax.fill_between(t_axis, sir_result['i_curve'],
                    alpha=0.15, color=COLOR_LOG4SHELL)
    pk_sir = sir_result['peak_infected']
    ax.axvline(sir_result['peak_time'], color='gray', ls='--', lw=1.2)
    ax.text(
        0.97, 0.97,
        f'Peak: {pk_sir:.0f} ({pk_sir/N_sir*100:.1f}%)',
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    )
    ax.set_xlabel('Time Step t', fontsize=9)
    ax.set_ylabel('Infected I(t)', fontsize=9)
    ax.set_title('SIR — Log4Shell Seed (β=0.3)', fontsize=11,
                 fontweight='bold')

    # ── PANEL [1,2]: Robustness Comparison (compact) ─────────────
    ax = axes[1, 2]
    f_r, s_r = zip(*random_results)
    f_t, s_t = zip(*targeted_results)
    ax.plot(f_r, s_r, color=COLOR_NEUTRAL, ls='--', lw=2.0,
            label='Random')
    ax.plot(f_t, s_t, color=COLOR_LOG4SHELL, ls='-', lw=2.0,
            label='Targeted')
    ax.axhline(0.05, color='gray', ls=':', lw=1.0)
    ax.set_xlabel('Fraction removed', fontsize=9)
    ax.set_ylabel('S/N', fontsize=9)
    ax.set_title('Robustness: Random vs Targeted', fontsize=11,
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    # ── Save ──────────────────────────────────────────────────────
    plt.tight_layout(pad=2.5)
    plt.savefig('outputs/SUMMARY_DASHBOARD.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[DASHBOARD] Summary dashboard saved ✓")
