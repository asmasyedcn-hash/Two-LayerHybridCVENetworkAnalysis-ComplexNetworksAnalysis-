"""
community_detection.py
======================
MITS 6700G Group 8 — Module 5: Community Detection

Louvain community detection on the global CVE projection.
Identifies the community containing Log4Shell (CVE-2021-44228)
and produces publication-quality visualisations.

Functions:
    detect_communities_louvain(G_cve)
    plot_community_size_distribution(partition, modularity, G_cve)
    visualize_communities_network(G_cve, partition, modularity, max_nodes)
    log4shell_community_report(G_cve, partition, df)
"""

import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
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


def detect_communities_louvain(G_cve):
    """
    Detect communities using the Louvain algorithm.

    Uses python-louvain (community module) with random_state=42
    for reproducibility.

    Parameters
    ----------
    G_cve : nx.Graph
        CVE-CVE projection (Layer 1 global graph).

    Returns
    -------
    tuple[dict, float]
        (partition, modularity) — node→community mapping and Q score.
    """
    import community as community_louvain

    partition  = community_louvain.best_partition(G_cve, random_state=42)
    modularity = community_louvain.modularity(partition, G_cve)

    sizes  = Counter(partition.values())
    n_comm = len(sizes)

    print(f"[COMMUNITY] Communities detected : {n_comm}")
    print(f"[COMMUNITY] Modularity Q         : {modularity:.4f}")
    print(f"[COMMUNITY] Largest community    : {max(sizes.values())} nodes")
    print(f"[COMMUNITY] Smallest community   : {min(sizes.values())} nodes")
    top5 = sorted(sizes.items(), key=lambda x: -x[1])[:5]
    print(f"[COMMUNITY] Top 5 by size        : {top5}")

    return partition, modularity


def plot_community_size_distribution(partition, modularity, G_cve):
    """
    Bar chart of the top-20 community sizes with count and percentage
    of the total network (N) annotated on each bar.

    FIX-4 APPLIED: G_cve is passed as parameter; N is derived from it.

    Parameters
    ----------
    partition : dict
        Node→community mapping from detect_communities_louvain.
    modularity : float
        Louvain modularity Q score.
    G_cve : nx.Graph
        The global CVE-CVE graph (used to compute N).
    """
    # FIX-4: N defined from G_cve
    N = G_cve.number_of_nodes()

    sizes      = Counter(partition.values())
    log4_comm  = partition.get('CVE-2021-44228', -1)
    sorted_comms = sorted(sizes.items(), key=lambda x: -x[1])
    top_comms    = sorted_comms[:20]

    fig, ax = plt.subplots(figsize=(14, 7))
    comm_ids   = [c for c, _ in top_comms]
    comm_sizes = [s for _, s in top_comms]
    colors = [
        COLOR_LOG4SHELL if c == log4_comm else PALETTE_COMM[i % 20]
        for i, c in enumerate(comm_ids)
    ]

    bars = ax.bar(
        range(len(top_comms)), comm_sizes,
        color=colors, edgecolor='white', linewidth=0.8
    )

    # FIX-4: Annotate with count AND % of N
    for bar, (comm_id, size) in zip(bars, top_comms):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(comm_sizes) * 0.01,
            f'{size}\n({size/N*100:.1f}%)',
            ha='center', va='bottom', fontsize=8, fontweight='bold'
        )

    # Annotate the Log4Shell community bar
    if log4_comm in comm_ids:
        idx = comm_ids.index(log4_comm)
        ax.annotate(
            '← Log4Shell\n   community',
            xy=(idx, sizes[log4_comm]),
            xytext=(idx + 1.5, sizes[log4_comm] * 1.1),
            fontsize=9, color=COLOR_LOG4SHELL, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLOR_LOG4SHELL)
        )

    ax.set_xticks(range(len(top_comms)))
    ax.set_xticklabels([f'C{c}' for c in comm_ids], fontsize=9)
    ax.set_xlabel('Community ID (sorted by size)')
    ax.set_ylabel('Number of CVEs')
    ax.set_title(
        f'Community Size Distribution (Louvain Algorithm)\n'
        f'Q = {modularity:.4f} | {len(sizes)} communities | '
        f'N = {N} total nodes',
        fontweight='bold'
    )

    log4_p  = mpatches.Patch(color=COLOR_LOG4SHELL,
                               label='Log4Shell community')
    other_p = mpatches.Patch(color=PALETTE_COMM[0],
                               label='Other communities')
    ax.legend(handles=[log4_p, other_p], fontsize=10)

    plt.tight_layout(pad=2.0)
    plt.savefig('outputs/community_size_dist.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[COMMUNITY] community_size_dist.png saved ✓")


def visualize_communities_network(G_cve, partition, modularity,
                                   max_nodes=600):
    """
    Publication-quality network graph coloured by community membership.

    If the graph exceeds max_nodes, a random sample is drawn while
    always preserving CVE-2021-44228. Layout computed with spring
    layout (seed=42). Only Log4Shell and the top-5 degree nodes
    are labelled to avoid clutter.

    Parameters
    ----------
    G_cve : nx.Graph
        Full global CVE-CVE graph.
    partition : dict
        Node→community mapping.
    modularity : float
        Louvain modularity Q score (used in title).
    max_nodes : int, optional
        Maximum nodes to display (default: 600).
    """
    N_full    = G_cve.number_of_nodes()
    seed_node = 'CVE-2021-44228'

    if N_full > max_nodes:
        rnd.seed(42)
        other_nodes   = [n for n in G_cve.nodes() if n != seed_node]
        sampled       = rnd.sample(other_nodes,
                                   min(max_nodes - 1, len(other_nodes)))
        sampled_nodes = set(sampled) | {seed_node}
        G_plot        = G_cve.subgraph(sampled_nodes).copy()
        print(f"[COMM_VIZ] Sampling {len(sampled_nodes)} of {N_full} nodes")
    else:
        G_plot = G_cve

    pos = nx.spring_layout(
        G_plot,
        k=2.5 / np.sqrt(G_plot.number_of_nodes()),
        iterations=100,
        seed=42
    )

    comm_ids    = [partition.get(n, 0) for n in G_plot.nodes()]
    unique_c    = list(set(comm_ids))
    cmap_dict   = {c: PALETTE_COMM[i % 20] for i, c in enumerate(unique_c)}
    node_colors = [cmap_dict[c] for c in comm_ids]
    degrees     = dict(G_plot.degree())
    max_deg     = max(degrees.values()) if degrees else 1
    node_sizes  = [60 + 220 * (degrees[n] / max_deg) for n in G_plot.nodes()]

    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_facecolor('#F8F9FA')

    nx.draw_networkx_edges(
        G_plot, pos, ax=ax,
        alpha=0.12, edge_color='#AAAAAA', width=0.5
    )
    nx.draw_networkx_nodes(
        G_plot, pos, ax=ax,
        node_color=node_colors, node_size=node_sizes,
        alpha=0.82, linewidths=0.4, edgecolors='white'
    )

    # Log4Shell — always on top, larger, red
    if seed_node in G_plot.nodes():
        nx.draw_networkx_nodes(
            G_plot, pos, ax=ax,
            nodelist=[seed_node],
            node_color=COLOR_LOG4SHELL,
            node_size=900, alpha=1.0,
            linewidths=2.5, edgecolors='black'
        )
        nx.draw_networkx_labels(
            G_plot, pos, ax=ax,
            labels={seed_node: 'Log4Shell\nCVE-2021-44228'},
            font_size=8, font_weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLOR_LOG4SHELL, alpha=0.92)
        )

    # Label top-5 degree nodes (excluding Log4Shell)
    top5 = sorted(degrees, key=degrees.get, reverse=True)
    top5 = [n for n in top5 if n != seed_node][:5]
    nx.draw_networkx_labels(
        G_plot, pos, ax=ax,
        labels={n: n.replace('CVE-', '') for n in top5},
        font_size=7, font_color='#333333'
    )

    # Legend — top 10 communities + Log4Shell patch
    patches = [
        mpatches.Patch(color=cmap_dict[c], label=f'Community {c}')
        for c in sorted(unique_c)[:10]
    ]
    patches.append(
        mpatches.Patch(color=COLOR_LOG4SHELL,
                       label='Log4Shell (CVE-2021-44228)')
    )
    ax.legend(
        handles=patches, loc='upper left', fontsize=9,
        title='Communities', title_fontsize=10,
        framealpha=0.92, ncol=2
    )

    ax.set_title(
        f'CVE Vulnerability Network — Community Structure\n'
        f'Louvain | Q = {modularity:.4f} | '
        f'{len(unique_c)} communities | '
        f'N shown = {G_plot.number_of_nodes()}',
        fontsize=15, fontweight='bold', pad=20
    )
    ax.axis('off')
    plt.tight_layout(pad=1.5)
    plt.savefig('outputs/community_network_plot.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[COMMUNITY] community_network_plot.png saved ✓")


def log4shell_community_report(G_cve, partition, df):
    """
    Print a formatted report for Log4Shell's community and export
    its member CVEs to CSV.

    Parameters
    ----------
    G_cve : nx.Graph
        Global CVE-CVE graph.
    partition : dict
        Node→community mapping.
    df : pd.DataFrame
        Full parsed CVE DataFrame (for CVSS/severity lookup).
    """
    log4_comm = partition.get('CVE-2021-44228', None)
    if log4_comm is None:
        print("[L4S_COMM] Log4Shell not found in partition — skipping report")
        return

    members = [n for n, c in partition.items() if c == log4_comm]

    # Merge with CVE metadata
    df_members = pd.DataFrame({'cve_id': members})
    df_members = df_members.merge(
        df[['cve_id', 'cvss_score', 'severity']], on='cve_id', how='left'
    ).sort_values('cvss_score', ascending=False)

    mean_cvss   = df_members['cvss_score'].mean()
    pct_critical = (
        (df_members['cvss_score'] >= 9.0).sum() / len(df_members) * 100
        if len(df_members) > 0 else 0.0
    )

    print(f"[L4S_COMM] Log4Shell community ID  : {log4_comm}")
    print(f"[L4S_COMM] Community size          : {len(members)} CVEs")
    print(f"[L4S_COMM] Mean CVSS in community  : {mean_cvss:.2f}")
    print(f"[L4S_COMM] % Critical in community : {pct_critical:.1f}%")
    print(f"[L4S_COMM] Top 10 by CVSS:")

    top10_members = df_members.head(10)
    for i, row in enumerate(top10_members.itertuples(), 1):
        cvss_val = row.cvss_score if not pd.isna(row.cvss_score) else 0.0
        severity = row.severity   if not pd.isna(row.severity)   else 'UNKNOWN'
        print(f"[L4S_COMM]   {i}. {row.cve_id}  {cvss_val:.1f}  {severity}")

    df_members.to_csv('outputs/log4shell_community_members.csv', index=False)
    print("[L4S_COMM] Saved: outputs/log4shell_community_members.csv")
