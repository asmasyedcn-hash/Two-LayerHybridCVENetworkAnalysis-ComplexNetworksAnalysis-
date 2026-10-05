"""
main.py
=======
MITS 6700G — Complex Networks Final Project
Group 8: Vulnerability Dependency Network Analysis
Two-Layer Hybrid Approach | Final Version (All fixes applied)

Layer 1 (Global): All CVEs with CVSS >= 7.0
                  Betweenness: k=1000 FIXED, seed=42
Layer 2 (Local):  2-hop ego-network of CVE-2021-44228
                  Betweenness: EXACT (small graph)

Execution order:
  1. pip install -r requirements.txt
  2. python download_data.py
  3. python main.py

Data source: fkie-cad/nvd-json-data-feeds (Fraunhofer FKIE)
             NVD API 2.0 mirror — identical CVE records to NIST.
"""

import os
import sys
import time

os.makedirs('outputs', exist_ok=True)

print("=" * 65)
print("  MITS 6700G — CVE Network Analysis Pipeline")
print("  Group 8 | Two-Layer Hybrid | k=1000 Fixed")
print("=" * 65)

# ── STEP 0: Parse and verify ──────────────────────────────────────
from parse_nvd import parse_nvd_json, filter_cves_global, verify_log4shell

# DATA SOURCE: fkie-cad/nvd-json-data-feeds (Fraunhofer FKIE)
# Real NVD CVE data mirrored from official NVD API 2.0
# File created by: python download_data.py  ← run that first
DATA_FILE = 'data/nvdcve-1.1-2021.json'
if not os.path.exists(DATA_FILE):
    print("[FATAL] data/nvdcve-1.1-2021.json not found!")
    print("[FATAL] Run:  python download_data.py  first.")
    sys.exit(1)

t0 = time.time()
df_raw   = parse_nvd_json(DATA_FILE)
ok       = verify_log4shell(df_raw)
if not ok:
    print("[FATAL] Log4Shell not found. Stopping.")
    sys.exit(1)
df_global_raw = filter_cves_global(df_raw)

# ── STEP 1: Build graphs ──────────────────────────────────────────
print("\n" + "─" * 65)
print("  STEP 1: Building graphs")
print("─" * 65)

from build_graph import (build_bipartite_graph, build_cve_projection,
                          build_log4shell_ego_graph, graph_basic_stats)

G_bip    = build_bipartite_graph(df_global_raw)   # Layer 1 bipartite
G_global = build_cve_projection(G_bip)             # Layer 1 CVE-CVE (LCC)
G_local  = build_log4shell_ego_graph(G_global)     # Layer 2 ego-graph

if G_local is None:
    print("[FATAL] Could not build ego-graph. Log4Shell not in LCC.")
    sys.exit(1)

print("\n── LAYER 1: GLOBAL GRAPH ──")
stats_g = graph_basic_stats(G_global, "Global CVE Projection (Layer 1)")
print("\n── LAYER 2: LOCAL EGO-GRAPH ──")
stats_l = graph_basic_stats(G_local,  "Log4Shell Ego-Graph (Layer 2)")

# ── STEP 2: Degree distribution ───────────────────────────────────
print("\n" + "─" * 65)
print("  STEP 2: Degree distribution analysis")
print("─" * 65)

from degree_dist import plot_degree_distribution

gamma_g = plot_degree_distribution(G_global, 'global',
                                    'Layer 1 | CVSS>=7.0 | 2021')
gamma_l = plot_degree_distribution(G_local,  'local',
                                    'Layer 2 | 2-hop Log4Shell ego')

# ── STEP 3: Centrality (k=1000 FIXED for Layer 1) ─────────────────
print("\n" + "─" * 65)
print("  STEP 3: Centrality analysis")
print("─" * 65)

from centrality import (compute_all_centrality, plot_top10_centrality,
                         report_log4shell_centrality, plot_centrality_radar)

df_gcent, df_lcent = compute_all_centrality(G_global, G_local, df_raw)

for metric in ['pagerank', 'betweenness', 'degree_cent', 'closeness']:
    plot_top10_centrality(df_gcent, metric, 'Layer 1 Global', 'global')
    plot_top10_centrality(df_lcent, metric, 'Layer 2 Local',  'local')

report_log4shell_centrality(df_gcent, df_lcent)
plot_centrality_radar(df_lcent)

# ── STEP 4: Community detection ───────────────────────────────────
print("\n" + "─" * 65)
print("  STEP 4: Community detection")
print("─" * 65)

from community_detection import (detect_communities_louvain,
                                   plot_community_size_distribution,
                                   visualize_communities_network,
                                   log4shell_community_report)

partition, modularity = detect_communities_louvain(G_global)
plot_community_size_distribution(partition, modularity, G_global)  # FIX-4
visualize_communities_network(G_global, partition, modularity, max_nodes=600)
log4shell_community_report(G_global, partition, df_raw)

# ── STEP 5: SIR epidemic model ────────────────────────────────────
print("\n" + "─" * 65)
print("  STEP 5: SIR epidemic simulation")
print("─" * 65)

from sir_model import (sir_simulation, plot_sir_single,
                        compare_beta_values, hop_reachability_plot)

beta_values = [0.1, 0.2, 0.3, 0.5]
for beta in beta_values:
    res = sir_simulation(G_local, 'CVE-2021-44228',
                          beta=beta, gamma_rate=0.1,
                          max_steps=100, n_runs=20)
    if res is not None:
        plot_sir_single(res, beta, 0.1,
                        'Layer 2 — Log4Shell Ego-Graph')

compare_beta_values(G_local, 'CVE-2021-44228', beta_values,
                    gamma_rate=0.1)
hop_reachability_plot(G_global, 'CVE-2021-44228', max_hops=4)

# ── STEP 6: Robustness ────────────────────────────────────────────
print("\n" + "─" * 65)
print("  STEP 6: Robustness analysis")
print("─" * 65)

from robustness import (random_attack_simulation,
                         targeted_attack_simulation,
                         plot_robustness_comparison)

r_results = random_attack_simulation(G_global,  n_steps=30)
t_results = targeted_attack_simulation(G_global, n_steps=30)
plot_robustness_comparison(r_results, t_results,
                            G_global.number_of_nodes())

# ── STEP 7: Summary dashboard ─────────────────────────────────────
print("\n" + "─" * 65)
print("  STEP 7: Summary dashboard")
print("─" * 65)

from summary_dashboard import plot_summary_dashboard

sir_main = sir_simulation(G_local, 'CVE-2021-44228',
                           beta=0.3, gamma_rate=0.1,
                           max_steps=100, n_runs=20)

plot_summary_dashboard(
    df_global_cent   = df_gcent,
    df_raw           = df_raw,
    partition        = partition,
    modularity       = modularity,
    sir_result       = sir_main,
    n_communities    = len(set(partition.values())),
    gamma_exp        = gamma_g,
    random_results   = r_results,
    targeted_results = t_results,
)

# ── Done ──────────────────────────────────────────────────────────
elapsed = time.time() - t0
print("\n" + "=" * 65)
print(f"  ✅  ALL ANALYSIS COMPLETE  ({elapsed:.0f}s)")
print(f"  📁  All outputs saved to: outputs/")
print("=" * 65)

files = os.listdir('outputs/')
print(f"\n  Files generated ({len(files)}):")
for fname in sorted(files):
    kb = os.path.getsize(f'outputs/{fname}') / 1024
    print(f"    {fname:<50}  {kb:>8.1f} KB")
