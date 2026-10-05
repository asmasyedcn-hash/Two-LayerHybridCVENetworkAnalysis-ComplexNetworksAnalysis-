"""
build_graph.py
==============
MITS 6700G Group 8 — Module 2: Graph Construction

Builds the two-layer hybrid graph structure:
  Layer 1 (Global): Bipartite CVE↔CPE graph → projected to CVE-CVE
  Layer 2 (Local):  2-hop ego-network centred on CVE-2021-44228

Functions:
    build_bipartite_graph(df)            — CVE↔CPE bipartite graph
    build_cve_projection(G)              — Project onto CVE nodes (LCC)
    build_log4shell_ego_graph(G_cve)     — 2-hop ego-network
    graph_basic_stats(G, label)          — Full diagnostic stats box
"""

import networkx as nx
import numpy as np
import pandas as pd

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


def build_bipartite_graph(df):
    """
    Build a CVE ↔ CPE bipartite graph from the filtered Layer 1 data.

    CVE nodes (bipartite=0) are linked to CPE product nodes
    (bipartite=1) via edges representing 'affects' relationships.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered CVE DataFrame (output of filter_cves_global).

    Returns
    -------
    nx.Graph
        Bipartite graph with CVE and CPE nodes.
    """
    G = nx.Graph()

    for row in df.itertuples():
        # Add CVE node with attributes
        G.add_node(
            row.cve_id,
            bipartite=0,
            node_type='CVE',
            cvss=row.cvss_score,
            severity=row.severity,
            published=row.published_date,
        )
        # Add CPE nodes and edges
        for product in row.affected_products:
            G.add_node(product, bipartite=1, node_type='CPE')
            G.add_edge(row.cve_id, product)

    # Verify degree sum == 2 * edges
    sum_deg = sum(dict(G.degree()).values())
    verify  = "PASSED" if sum_deg == 2 * G.number_of_edges() else "FAILED"

    cve_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 0]
    cpe_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 1]

    print(f"[BUILD_BIPARTITE] Bipartite graph built:")
    print(f"[BUILD_BIPARTITE]   CVE nodes  : {len(cve_nodes)} (bipartite=0)")
    print(f"[BUILD_BIPARTITE]   CPE nodes  : {len(cpe_nodes)} (bipartite=1)")
    print(f"[BUILD_BIPARTITE]   Total edges: {G.number_of_edges()}")
    print(f"[BUILD_BIPARTITE]   Degree sum = 2 * edges? {verify}")

    return G


def build_cve_projection(G):
    """
    Project the bipartite graph onto CVE nodes only.

    Two CVEs are connected if they share at least one common CPE product.
    Always returns the Largest Connected Component (LCC) for analysis.

    Parameters
    ----------
    G : nx.Graph
        Bipartite CVE↔CPE graph from build_bipartite_graph().

    Returns
    -------
    nx.Graph
        CVE-CVE projection restricted to the LCC.
    """
    from networkx.algorithms import bipartite

    cve_nodes = {n for n, d in G.nodes(data=True) if d.get('bipartite') == 0}
    G_cve     = bipartite.projected_graph(G, cve_nodes)

    # Copy node attributes from bipartite graph to projection
    for node in G_cve.nodes():
        if node in G.nodes:
            G_cve.nodes[node].update(G.nodes[node])

    # Extract Largest Connected Component
    components = list(nx.connected_components(G_cve))
    lcc_nodes  = max(components, key=len)
    G_lcc      = G_cve.subgraph(lcc_nodes).copy()
    lcc_pct    = len(lcc_nodes) / G_cve.number_of_nodes() * 100

    print(f"[BUILD_PROJECTION] CVE-CVE projection:")
    print(f"[BUILD_PROJECTION]   Total CVE nodes: {G_cve.number_of_nodes()}")
    print(f"[BUILD_PROJECTION]   Total edges    : {G_cve.number_of_edges()}")
    print(f"[BUILD_PROJECTION]   Components     : {len(components)}")
    print(f"[BUILD_PROJECTION]   LCC size       : {len(lcc_nodes)} nodes "
          f"({lcc_pct:.1f}% of projection)")
    print(f"[BUILD_PROJECTION]   ← Returning LCC only for all analyses")

    return G_lcc


def build_log4shell_ego_graph(G_cve, hops=2):
    """
    Extract the 2-hop ego-network centred on CVE-2021-44228 for
    LAYER 2 (LOCAL) analysis.

    Parameters
    ----------
    G_cve : nx.Graph
        CVE-CVE projection (LCC) from build_cve_projection().
    hops : int, optional
        Radius of the ego-network (default: 2).

    Returns
    -------
    nx.Graph or None
        Ego-graph subgraph, or None if Log4Shell is not in G_cve.
    """
    seed = 'CVE-2021-44228'

    if seed not in G_cve:
        print(f"[EGO_GRAPH] ✗ Log4Shell not in graph — check LCC extraction")
        return None

    G_ego   = nx.ego_graph(G_cve, seed, radius=hops).copy()
    density = nx.density(G_ego)

    print(f"[EGO_GRAPH] Layer 2 (Local) node selection basis:")
    print(f"[EGO_GRAPH]   Rule: Within {hops} hops of CVE-2021-44228")
    print(f"[EGO_GRAPH]   Nodes : {G_ego.number_of_nodes()}  "
          f"(all CVEs reachable in <= {hops} hops)")
    print(f"[EGO_GRAPH]   Edges : {G_ego.number_of_edges()}")
    print(f"[EGO_GRAPH]   Density: {density:.4f}")

    return G_ego


def graph_basic_stats(G, label="Graph"):
    """
    Compute and print a full diagnostic statistics box for any graph.

    Parameters
    ----------
    G : nx.Graph
        Any NetworkX graph.
    label : str, optional
        Display name for the graph in the output box.

    Returns
    -------
    dict
        Dictionary of all computed statistics.
    """
    N       = G.number_of_nodes()
    M       = G.number_of_edges()
    degrees = [d for _, d in G.degree()]

    avg_deg = np.mean(degrees) if degrees else 0.0
    std_deg = np.std(degrees)  if degrees else 0.0
    max_deg = max(degrees)     if degrees else 0
    min_deg = min(degrees)     if degrees else 0

    max_node = max(G.degree(), key=lambda x: x[1])[0] if N > 0 else 'N/A'

    density    = nx.density(G)
    components = list(nx.connected_components(G))
    n_comps    = len(components)
    lcc_size   = len(max(components, key=len)) if components else 0
    lcc_pct    = lcc_size / N * 100 if N > 0 else 0.0

    # Clustering — skip for dense graphs (too slow); compute exactly only for small graphs
    if M <= 10000:
        avg_clust = nx.average_clustering(G)
    else:
        avg_clust = float('nan')  # skip — graph too dense for clustering computation

    # Degree verification
    sum_deg    = sum(degrees)
    deg_verify = "PASSED" if sum_deg == 2 * M else "FAILED"
    is_directed = "YES" if G.is_directed() else "NO"

    import math
    clust_str = f"{avg_clust:.4f}" if not math.isnan(avg_clust) else "N/A (graph too dense)"
    print(f"  ┌─────────────────────────────────────────────────┐")
    print(f"  │  GRAPH STATISTICS: {label}")
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │  Nodes (N)             : {N}")
    print(f"  │  Edges (M)             : {M}")
    print(f"  │  Directed              : {is_directed}")
    print(f"  │  Average degree        : {avg_deg:.2f}")
    print(f"  │  Degree std deviation  : {std_deg:.2f}")
    print(f"  │  Max degree            : {max_deg}  (node: {max_node})")
    print(f"  │  Min degree            : {min_deg}")
    print(f"  │  Graph density         : {density:.6f}")
    print(f"  │  Connected components  : {n_comps}")
    print(f"  │  LCC size              : {lcc_size} nodes ({lcc_pct:.1f}%)")
    print(f"  │  Average clustering    : {clust_str}")
    print(f"  │  Degree verification   : {deg_verify}")
    print(f"  │    sum(degrees)=2*M?   : {sum_deg} = {2*M}")
    print(f"  └─────────────────────────────────────────────────┘")

    return {
        'N': N, 'M': M, 'avg_deg': avg_deg, 'std_deg': std_deg,
        'max_deg': max_deg, 'min_deg': min_deg, 'density': density,
        'n_components': n_comps, 'lcc_size': lcc_size, 'lcc_pct': lcc_pct,
        'avg_clustering': avg_clust, 'degree_verify': deg_verify,
    }
