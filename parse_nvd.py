"""
parse_nvd.py
============
MITS 6700G Group 8 — Module 1: NVD JSON Data Parsing

Parses the NVD CVE 2021 JSON feed (legacy 1.1 schema) from
fkie-cad/nvd-json-data-feeds into a clean pandas DataFrame.

Functions:
    parse_nvd_json(filepath)      — Load & parse full dataset
    filter_cves_global(df)        — Layer 1 filter: CVSS >= 7.0
    verify_log4shell(df)          — Confirm Log4Shell is present
"""

import json
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — must be FIRST
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

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
COLOR_LOG4SHELL = '#E63946'   # Vivid red  — Log4Shell always this
COLOR_HIGHLIGHT = '#F4A261'   # Orange     — secondary highlight
COLOR_NEUTRAL   = '#457B9D'   # Steel blue — regular nodes/bars
COLOR_GOOD      = '#2A9D8F'   # Teal       — low risk
COLOR_BAD       = '#E76F51'   # Coral      — high risk


def parse_nvd_json(filepath):
    """
    Load and parse an NVD JSON feed file (legacy 1.1 schema).

    Extracts CVE ID, CVSS score, severity, affected CPE products,
    and published date from each CVE_Item entry.

    Parameters
    ----------
    filepath : str
        Path to the NVD JSON file (e.g., data/nvdcve-1.1-2021.json).

    Returns
    -------
    pd.DataFrame
        Columns: cve_id, cvss_score, severity, affected_products,
                 published_date
    """
    print(f"[PARSE] Loading: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Support both legacy 1.1 key ('CVE_Items') and NVD API 2.0 key ('cve_items')
    items_list = data.get('CVE_Items') or data.get('cve_items', [])

    records = []
    for item in items_list:
        # ── CVE ID ────────────────────────────────────────────────
        # NVD API 2.0: item['id']
        # Legacy 1.1:  item['cve']['CVE_data_meta']['ID']
        if 'id' in item:
            cve_id = item['id']
        else:
            try:
                cve_id = item['cve']['CVE_data_meta']['ID']
            except (KeyError, TypeError):
                cve_id = 'UNKNOWN'

        # ── CVSS score & severity (V3.1 > V3.0 > V2 fallback) ────
        cvss_score = 0.0
        severity   = 'UNKNOWN'
        try:
            metrics = item.get('metrics', {})
            if metrics.get('cvssMetricV31'):
                cvss_data  = metrics['cvssMetricV31'][0]['cvssData']
                cvss_score = cvss_data['baseScore']
                severity   = cvss_data['baseSeverity']
            elif metrics.get('cvssMetricV30'):
                cvss_data  = metrics['cvssMetricV30'][0]['cvssData']
                cvss_score = cvss_data['baseScore']
                severity   = cvss_data['baseSeverity']
            elif metrics.get('cvssMetricV2'):
                cvss_data  = metrics['cvssMetricV2'][0]['cvssData']
                cvss_score = cvss_data['baseScore']
                severity   = metrics['cvssMetricV2'][0].get('baseSeverity', 'UNKNOWN')
            # Legacy 1.1 fallback
            elif item.get('impact', {}).get('baseMetricV3'):
                cvss_score = item['impact']['baseMetricV3']['cvssV3']['baseScore']
                severity   = item['impact']['baseMetricV3']['cvssV3']['baseSeverity']
        except (KeyError, TypeError, IndexError):
            pass

        # ── Affected CPE products ─────────────────────────────────
        affected_products = []
        try:
            configs = item.get('configurations', [])
            # NVD API 2.0: list of {nodes: [...]}
            if isinstance(configs, list):
                for conf in configs:
                    for node in conf.get('nodes', []):
                        for cpe in node.get('cpeMatch', []):
                            uri = cpe.get('criteria', '')
                            if uri:
                                affected_products.append(uri)
            # Legacy 1.1: dict with 'nodes' key
            elif isinstance(configs, dict):
                for node in configs.get('nodes', []):
                    for cpe in node.get('cpe_match', []):
                        uri = cpe.get('cpe23Uri', '')
                        if uri:
                            affected_products.append(uri)
                    for child in node.get('children', []):
                        for cpe in child.get('cpe_match', []):
                            uri = cpe.get('cpe23Uri', '')
                            if uri:
                                affected_products.append(uri)
        except (KeyError, TypeError):
            pass
        affected_products = list(set(affected_products))

        # ── Published date (YYYY-MM-DD only) ──────────────────────
        published_date = (item.get('published') or
                          item.get('publishedDate', ''))[:10]

        records.append({
            'cve_id':            cve_id,
            'cvss_score':        cvss_score,
            'severity':          severity,
            'affected_products': affected_products,
            'published_date':    published_date,
        })

    df = pd.DataFrame(records,
                      columns=['cve_id', 'cvss_score', 'severity',
                                'affected_products', 'published_date'])

    # ── Summary statistics ────────────────────────────────────────
    total        = len(df)
    with_cvss    = (df['cvss_score'] > 0).sum()
    no_cpe       = df['affected_products'].apply(len).eq(0).sum()
    n_critical   = (df['cvss_score'] >= 9.0).sum()
    n_high       = ((df['cvss_score'] >= 7.0) & (df['cvss_score'] < 9.0)).sum()
    n_medium     = ((df['cvss_score'] >= 4.0) & (df['cvss_score'] < 7.0)).sum()
    n_low        = (df['cvss_score'] < 4.0).sum()

    print(f"[PARSE] Total CVEs loaded        : {total}")
    print(f"[PARSE] CVEs with CVSS score     : {with_cvss}")
    print(f"[PARSE] CVEs with no CPE data    : {no_cpe}")
    print(f"[PARSE] Critical (>=9.0)         : {n_critical}")
    print(f"[PARSE] High (7.0-8.9)           : {n_high}")
    print(f"[PARSE] Medium (4.0-6.9)         : {n_medium}")
    print(f"[PARSE] Low (<4.0)               : {n_low}")

    return df


def filter_cves_global(df):
    """
    Filter CVEs for LAYER 1 (GLOBAL) analysis.

    Keeps only High + Critical CVEs (CVSS >= 7.0) that have at least
    one affected CPE product. This is the node selection basis for the
    global bipartite/projection graph.

    Parameters
    ----------
    df : pd.DataFrame
        Full parsed CVE DataFrame from parse_nvd_json().

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame for Layer 1 (Global) analysis.
    """
    total = len(df)

    print(f"[FILTER_GLOBAL] Layer 1 (Global) node selection basis:")
    print(f"[FILTER_GLOBAL]   Rule 1: CVSS score >= 7.0 (High + Critical only)")
    print(f"[FILTER_GLOBAL]   Rule 2: Must have >= 1 affected CPE product")

    df_global = df[
        (df['cvss_score'] >= 7.0) &
        (df['affected_products'].apply(len) >= 1)
    ].copy().reset_index(drop=True)

    kept    = len(df_global)
    dropped = total - kept

    print(f"[FILTER_GLOBAL]   CVEs kept   : {kept} of {total} total")
    print(f"[FILTER_GLOBAL]   CVEs dropped: {dropped}")

    return df_global


def verify_log4shell(df):
    """
    Confirm that CVE-2021-44228 (Log4Shell) exists in the dataset
    before any processing begins.

    Parameters
    ----------
    df : pd.DataFrame
        Parsed CVE DataFrame.

    Returns
    -------
    bool
        True if Log4Shell found, False otherwise.
    """
    if 'CVE-2021-44228' in df['cve_id'].values:
        row = df[df['cve_id'] == 'CVE-2021-44228'].iloc[0]
        print(f"[VERIFY] ✓ Log4Shell found in dataset")
        print(f"[VERIFY]   CVE ID     : CVE-2021-44228")
        print(f"[VERIFY]   CVSS Score : {row.cvss_score}")
        print(f"[VERIFY]   Severity   : {row.severity}")
        print(f"[VERIFY]   Products   : {len(row.affected_products)} affected")
        print(f"[VERIFY]   Published  : {row.published_date}")
        return True
    else:
        print(f"[VERIFY] ✗ FATAL: CVE-2021-44228 not found!")
        print(f"[VERIFY]   Run python download_data.py first to get the correct file.")
        print(f"[VERIFY]   Source: fkie-cad/nvd-json-data-feeds (CVE-2021.json.xz)")
        return False
