"""
=============================================================================
DATA 620 – Week 4 Assignment: Centrality Measurement
Congressional Cosponsorship Network Analysis
118th Congress – All Bill Types

Author: Candace Grant
CUNY School of Professional Studies – M.S. Data Science
Date: March 2026

Description:
    This script implements the 7-step data loading plan described in the
    assignment document. It retrieves member and bill cosponsorship data
    from the Congress.gov API (Library of Congress), constructs a weighted
    undirected cosponsorship network using NetworkX, computes degree
    centrality, and compares centrality distributions across party
    affiliation using the Kruskal-Wallis test and Dunn's post-hoc test.

Data Source: https://api.congress.gov (v3)
=============================================================================
"""

# =============================================================================
# STEP 1 – ENVIRONMENT SETUP
# =============================================================================
import os
import time
import json
import warnings
import requests
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from itertools import combinations
from collections import defaultdict
from scipy import stats

# Optional: Dunn's post-hoc test (install with: pip install scikit-posthocs)
try:
    import scikit_posthocs as sp
    HAS_POSTHOCS = True
except ImportError:
    HAS_POSTHOCS = False
    print("Note: scikit-posthocs not installed. Dunn's post-hoc test will be skipped.")
    print("Install with: pip install scikit-posthocs\n")

warnings.filterwarnings('ignore')



# ─────────────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("CONGRESS_API_KEY", "PASTE_YOUR_KEY_HERE")

if API_KEY == "PASTE_YOUR_KEY_HERE":
    print("=" * 70)
    print("  WARNING: No API key detected!")
    print("  Please set your Congress.gov API key before running.")
    print("  Sign up free at: https://api.congress.gov/sign-up/")
    print("  Then either:")
    print('    export CONGRESS_API_KEY="your_key"')
    print("    OR edit API_KEY in this script directly.")
    print("=" * 70)
    print()

BASE_URL = "https://api.congress.gov/v3"
CONGRESS = 118
RATE_LIMIT_DELAY = 0.75  # seconds between API calls (conservative for 5000/hr limit)

# ── Region Mapping for Geographic Analysis ───────────────────────────────────
REGION_MAP = {
    # Northeast
    'CT': 'Northeast', 'ME': 'Northeast', 'MA': 'Northeast', 'NH': 'Northeast',
    'RI': 'Northeast', 'VT': 'Northeast', 'NJ': 'Northeast', 'NY': 'Northeast',
    'PA': 'Northeast',
    # South
    'DE': 'South', 'FL': 'South', 'GA': 'South', 'MD': 'South', 'NC': 'South',
    'SC': 'South', 'VA': 'South', 'WV': 'South', 'AL': 'South', 'KY': 'South',
    'MS': 'South', 'TN': 'South', 'AR': 'South', 'LA': 'South', 'OK': 'South',
    'TX': 'South', 'DC': 'South',
    # Midwest
    'IL': 'Midwest', 'IN': 'Midwest', 'MI': 'Midwest', 'OH': 'Midwest',
    'WI': 'Midwest', 'IA': 'Midwest', 'KS': 'Midwest', 'MN': 'Midwest',
    'MO': 'Midwest', 'NE': 'Midwest', 'ND': 'Midwest', 'SD': 'Midwest',
    # West
    'AZ': 'West', 'CO': 'West', 'ID': 'West', 'MT': 'West', 'NV': 'West',
    'NM': 'West', 'UT': 'West', 'WY': 'West', 'AK': 'West', 'CA': 'West',
    'HI': 'West', 'OR': 'West', 'WA': 'West',
    # Territories
    'AS': 'Territory', 'GU': 'Territory', 'MP': 'Territory', 'PR': 'Territory',
    'VI': 'Territory'
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def api_get(endpoint, params=None):
    """
    Make a GET request to the Congress.gov API with rate limiting and error handling.

    Parameters
    ----------
    endpoint : str
        API endpoint path (appended to BASE_URL).
    params : dict, optional
        Additional query parameters.

    Returns
    -------
    dict or None
        JSON response data, or None if the request failed.
    """
    if params is None:
        params = {}
    params['api_key'] = API_KEY
    params['format'] = 'json'

    url = f"{BASE_URL}/{endpoint}"

    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            print("  Rate limited – waiting 60 seconds...")
            time.sleep(60)
            return api_get(endpoint, params)  # Retry
        else:
            print(f"  API Error {response.status_code}: {url}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"  Request failed: {e}")
        return None
    finally:
        time.sleep(RATE_LIMIT_DELAY)


def paginate_api(endpoint, data_key, params=None, max_records=None):
    """
    Retrieve all records from a paginated API endpoint.

    Parameters
    ----------
    endpoint : str
        API endpoint path.
    data_key : str
        Key in the JSON response containing the list of records.
    params : dict, optional
        Additional query parameters.
    max_records : int, optional
        Maximum number of records to retrieve (None = all).

    Returns
    -------
    list
        All retrieved records.
    """
    if params is None:
        params = {}
    params['limit'] = 250  # Maximum allowed per request
    params['offset'] = 0

    all_records = []
    total_fetched = 0

    while True:
        data = api_get(endpoint, params)
        if data is None:
            break

        records = data.get(data_key, [])
        if not records:
            break

        all_records.extend(records)
        total_fetched += len(records)

        if max_records and total_fetched >= max_records:
            all_records = all_records[:max_records]
            break

        # Check for next page
        pagination = data.get('pagination', {})
        next_url = pagination.get('next', None)
        if next_url is None:
            break

        params['offset'] += 250
        print(f"    Fetched {total_fetched} records so far...")

    return all_records


# =============================================================================
# STEP 2 – RETRIEVE THE MEMBER LIST (NODE DATA)
# =============================================================================
def retrieve_members():
    """
    Retrieve all members of the 118th Congress from the Congress.gov API.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: bioguideId, name, party, state, chamber, district, region
    """
    print("=" * 70)
    print("STEP 2: Retrieving Member List (Node Data)")
    print("=" * 70)

    members_raw = paginate_api(
        f"member",
        data_key='members',
        params={'currentMember': 'true'}
    )

    print(f"  Retrieved {len(members_raw)} total current members.")

    # Parse into structured records
    members = []
    for m in members_raw:
        bioguide = m.get('bioguideId', '')
        name = m.get('name', '')
        state = m.get('state', '')
        party = m.get('partyName', 'Unknown')
        district = m.get('district', None)

        # Determine chamber from terms
        terms = m.get('terms', {}).get('item', [])
        chamber = 'Unknown'
        if terms:
            # Use the most recent term
            latest_term = terms[-1] if isinstance(terms, list) else terms
            chamber = latest_term.get('chamber', 'Unknown')

        # Map to region
        region = REGION_MAP.get(state, 'Unknown')

        members.append({
            'bioguideId': bioguide,
            'name': name,
            'party': party,
            'state': state,
            'chamber': chamber,
            'district': district,
            'region': region
        })

    members_df = pd.DataFrame(members)

    # Standardize party names
    party_map = {
        'Democratic': 'Democrat',
        'Republican': 'Republican',
        'Independent': 'Independent',
        'Libertarian': 'Independent'
    }
    members_df['party'] = members_df['party'].map(
        lambda x: party_map.get(x, x)
    )

    print(f"\n  Member Summary:")
    print(f"  {'─' * 40}")
    print(f"  Total members: {len(members_df)}")
    print(f"  Party breakdown:")
    for party, count in members_df['party'].value_counts().items():
        print(f"    {party}: {count}")
    print(f"  Chamber breakdown:")
    for chamber, count in members_df['chamber'].value_counts().items():
        print(f"    {chamber}: {count}")
    print()

    return members_df


# =============================================================================
# STEP 3 – RETRIEVE BILLS AND THEIR COSPONSORS (EDGE DATA)
# =============================================================================
def retrieve_bill_cosponsors(bill_types=None, max_bills_per_type=None):
    """
    Retrieve bills and their cosponsors from the 118th Congress.

    Parameters
    ----------
    bill_types : list, optional
        Bill type codes to retrieve. Default: all major types.
    max_bills_per_type : int, optional
        Maximum bills to retrieve per type (None = all). Use for testing.

    Returns
    -------
    list of dict
        Each dict contains: bill_id, bill_type, sponsor_bioguide,
        cosponsor_bioguides (list).
    """
    print("=" * 70)
    print("STEP 3: Retrieving Bills and Cosponsors (Edge Data)")
    print("=" * 70)

    if bill_types is None:
        bill_types = ['hr', 's', 'hjres', 'sjres', 'hconres', 'sconres', 'hres', 'sres']

    all_bill_data = []
    total_api_calls = 0

    for btype in bill_types:
        print(f"\n  Retrieving {btype.upper()} bills...")

        # Get list of bills
        bills = paginate_api(
            f"bill/{CONGRESS}/{btype}",
            data_key='bills',
            max_records=max_bills_per_type
        )

        print(f"    Found {len(bills)} {btype.upper()} bills.")

        for i, bill in enumerate(bills):
            bill_number = bill.get('number', '')
            bill_id = f"{btype}{bill_number}-{CONGRESS}"

            # Get bill detail to find sponsor
            bill_detail = api_get(f"bill/{CONGRESS}/{btype}/{bill_number}")
            total_api_calls += 1

            if bill_detail is None:
                continue

            bill_info = bill_detail.get('bill', {})

            # Extract sponsor bioguide ID
            sponsors = bill_info.get('sponsors', [])
            sponsor_bioguide = None
            if sponsors:
                sponsor = sponsors[0] if isinstance(sponsors, list) else sponsors
                sponsor_bioguide = sponsor.get('bioguideId', None)

            # Get cosponsors
            cosponsors_data = paginate_api(
                f"bill/{CONGRESS}/{btype}/{bill_number}/cosponsors",
                data_key='cosponsors'
            )
            total_api_calls += 1

            cosponsor_bioguides = []
            for cs in cosponsors_data:
                cs_bioguide = cs.get('bioguideId', None)
                if cs_bioguide:
                    cosponsor_bioguides.append(cs_bioguide)

            if sponsor_bioguide or cosponsor_bioguides:
                all_bill_data.append({
                    'bill_id': bill_id,
                    'bill_type': btype,
                    'sponsor_bioguide': sponsor_bioguide,
                    'cosponsor_bioguides': cosponsor_bioguides,
                    'n_cosponsors': len(cosponsor_bioguides)
                })

            # Progress update every 50 bills
            if (i + 1) % 50 == 0:
                print(f"    Processed {i + 1}/{len(bills)} {btype.upper()} bills "
                      f"(~{total_api_calls} API calls so far)")

    print(f"\n  Bill Retrieval Summary:")
    print(f"  {'─' * 40}")
    print(f"  Total bills with sponsor/cosponsor data: {len(all_bill_data)}")
    print(f"  Total API calls made: ~{total_api_calls}")
    bills_with_cosponsors = sum(1 for b in all_bill_data if b['n_cosponsors'] > 0)
    print(f"  Bills with at least 1 cosponsor: {bills_with_cosponsors}")
    if all_bill_data:
        avg_cosponsors = np.mean([b['n_cosponsors'] for b in all_bill_data])
        max_cosponsors = max(b['n_cosponsors'] for b in all_bill_data)
        print(f"  Average cosponsors per bill: {avg_cosponsors:.1f}")
        print(f"  Maximum cosponsors on a single bill: {max_cosponsors}")
    print()

    return all_bill_data


# =============================================================================
# STEP 4 – CONSTRUCT THE EDGE LIST
# =============================================================================
def construct_edge_list(bill_data):
    """
    Construct a weighted edge list from bill cosponsorship data.

    For each bill, edges are formed between:
      - The sponsor and each cosponsor
      - All pairs of cosponsors (co-cosponsor ties)

    Edge weight = number of bills shared between a pair of legislators.

    Parameters
    ----------
    bill_data : list of dict
        Output from retrieve_bill_cosponsors().

    Returns
    -------
    pd.DataFrame
        Edge list with columns: source, target, weight
    """
    print("=" * 70)
    print("STEP 4: Constructing Edge List")
    print("=" * 70)

    edge_counter = defaultdict(int)
    bills_processed = 0

    for bill in bill_data:
        sponsor = bill['sponsor_bioguide']
        cosponsors = bill['cosponsor_bioguides']

        if not cosponsors:
            continue

        # All legislators involved in this bill
        all_legislators = []
        if sponsor:
            all_legislators.append(sponsor)
        all_legislators.extend(cosponsors)

        # Remove duplicates while preserving order
        seen = set()
        unique_legislators = []
        for leg in all_legislators:
            if leg not in seen:
                seen.add(leg)
                unique_legislators.append(leg)

        # Create edges between all pairs
        for leg_a, leg_b in combinations(unique_legislators, 2):
            # Use sorted tuple as key to ensure undirected consistency
            edge_key = tuple(sorted([leg_a, leg_b]))
            edge_counter[edge_key] += 1

        bills_processed += 1

    # Convert to DataFrame
    edges = []
    for (source, target), weight in edge_counter.items():
        edges.append({
            'source': source,
            'target': target,
            'weight': weight
        })

    edge_df = pd.DataFrame(edges)

    print(f"  Edge List Summary:")
    print(f"  {'─' * 40}")
    print(f"  Bills processed: {bills_processed}")
    print(f"  Unique edges (legislator pairs): {len(edge_df)}")
    if not edge_df.empty:
        print(f"  Weight statistics:")
        print(f"    Mean weight: {edge_df['weight'].mean():.2f}")
        print(f"    Median weight: {edge_df['weight'].median():.0f}")
        print(f"    Max weight: {edge_df['weight'].max()}")
        print(f"    Edges with weight > 5: {(edge_df['weight'] > 5).sum()}")
    print()

    return edge_df


# =============================================================================
# STEP 5 – BUILD THE NETWORKX GRAPH
# =============================================================================
def build_network(members_df, edge_df):
    """
    Build a weighted undirected NetworkX graph from members and edges.

    Parameters
    ----------
    members_df : pd.DataFrame
        Member data (nodes) with attributes.
    edge_df : pd.DataFrame
        Edge list with weights.

    Returns
    -------
    nx.Graph
        The cosponsorship network.
    """
    print("=" * 70)
    print("STEP 5: Building NetworkX Graph")
    print("=" * 70)

    G = nx.Graph()

    # Add nodes with attributes
    for _, row in members_df.iterrows():
        G.add_node(
            row['bioguideId'],
            name=row['name'],
            party=row['party'],
            state=row['state'],
            chamber=row['chamber'],
            region=row['region']
        )

    # Add weighted edges
    if not edge_df.empty:
        for _, row in edge_df.iterrows():
            # Only add edge if both nodes exist in the graph
            if row['source'] in G.nodes and row['target'] in G.nodes:
                G.add_edge(row['source'], row['target'], weight=row['weight'])

    print(f"  Network Summary:")
    print(f"  {'─' * 40}")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    if G.number_of_nodes() > 0:
        density = nx.density(G)
        print(f"  Density: {density:.4f}")
        connected = nx.is_connected(G)
        print(f"  Connected: {connected}")
        if not connected:
            components = list(nx.connected_components(G))
            print(f"  Connected components: {len(components)}")
            largest_cc = max(components, key=len)
            print(f"  Largest component size: {len(largest_cc)}")
        isolates = list(nx.isolates(G))
        print(f"  Isolated nodes (no cosponsorship edges): {len(isolates)}")
    print()

    return G


# =============================================================================
# STEP 6 – DATA VALIDATION AND CLEANING
# =============================================================================
def validate_and_clean(G, members_df):
    """
    Validate the network and clean any data issues.

    Parameters
    ----------
    G : nx.Graph
        The cosponsorship network.
    members_df : pd.DataFrame
        Member data.

    Returns
    -------
    nx.Graph
        Cleaned network.
    """
    print("=" * 70)
    print("STEP 6: Data Validation and Cleaning")
    print("=" * 70)

    issues_found = 0

    # Check 1: Remove self-loops (if any)
    self_loops = list(nx.selfloop_edges(G))
    if self_loops:
        print(f"  Removing {len(self_loops)} self-loops")
        G.remove_edges_from(self_loops)
        issues_found += len(self_loops)

    # Check 2: Verify all nodes have required attributes
    missing_party = [n for n in G.nodes if G.nodes[n].get('party', '') in ['', 'Unknown', None]]
    if missing_party:
        print(f"  Nodes with missing party attribute: {len(missing_party)}")
        issues_found += len(missing_party)

    # Check 3: Verify edge weights are positive integers
    bad_weights = [(u, v) for u, v, d in G.edges(data=True) if d.get('weight', 0) <= 0]
    if bad_weights:
        print(f"  Edges with invalid weights: {len(bad_weights)}")
        G.remove_edges_from(bad_weights)
        issues_found += len(bad_weights)

    # Check 4: Remove isolated nodes (legislators with no cosponsorship ties)
    isolates = list(nx.isolates(G))
    if isolates:
        print(f"  Isolated nodes (no edges): {len(isolates)}")
        # Keep them in the graph – they have degree centrality of 0
        # which is analytically meaningful

    # Check 5: Verify node count is reasonable for 118th Congress
    n_nodes = G.number_of_nodes()
    if n_nodes < 400 or n_nodes > 600:
        print(f"  WARNING: Node count ({n_nodes}) outside expected range (400-600)")
        issues_found += 1

    if issues_found == 0:
        print("  ✓ All validation checks passed – no issues found.")
    else:
        print(f"\n  Total issues found and addressed: {issues_found}")

    # Summary after cleaning
    print(f"\n  Cleaned Network:")
    print(f"  {'─' * 40}")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print()

    return G


# =============================================================================
# STEP 7 – COMPUTE DEGREE CENTRALITY AND AGGREGATE BY GROUP
# =============================================================================
def compute_and_analyze_centrality(G):
    """
    Compute degree centrality, aggregate by party, and perform statistical tests.

    Parameters
    ----------
    G : nx.Graph
        The validated cosponsorship network.

    Returns
    -------
    pd.DataFrame
        DataFrame with centrality scores and node attributes for each legislator.
    """
    print("=" * 70)
    print("STEP 7: Compute Degree Centrality and Analyze by Group")
    print("=" * 70)

    # ── 7a. Compute Degree Centrality ────────────────────────────────────────
    dc = nx.degree_centrality(G)

    # Build results DataFrame
    results = []
    for node, centrality in dc.items():
        attrs = G.nodes[node]
        results.append({
            'bioguideId': node,
            'name': attrs.get('name', ''),
            'party': attrs.get('party', 'Unknown'),
            'state': attrs.get('state', ''),
            'chamber': attrs.get('chamber', ''),
            'region': attrs.get('region', ''),
            'degree': G.degree(node),
            'weighted_degree': G.degree(node, weight='weight'),
            'degree_centrality': centrality
        })

    results_df = pd.DataFrame(results)

    # ── 7b. Descriptive Statistics by Party ──────────────────────────────────
    print("\n  Degree Centrality by Party Affiliation:")
    print(f"  {'─' * 60}")

    party_stats = results_df.groupby('party')['degree_centrality'].agg(
        ['count', 'mean', 'median', 'std', 'min', 'max']
    ).round(4)
    print(party_stats.to_string())

    # ── 7c. Top 10 Most Central Legislators ──────────────────────────────────
    print(f"\n  Top 10 Legislators by Degree Centrality:")
    print(f"  {'─' * 60}")
    top10 = results_df.nlargest(10, 'degree_centrality')[
        ['name', 'party', 'state', 'chamber', 'degree', 'degree_centrality']
    ]
    print(top10.to_string(index=False))

    # ── 7d. Kruskal-Wallis Test (Party Comparison) ───────────────────────────
    print(f"\n  Statistical Testing: Kruskal-Wallis Test")
    print(f"  {'─' * 60}")
    print("  H₀: Degree centrality distributions are identical across parties")
    print("  H₁: At least one party's distribution differs\n")

    party_groups = {}
    for party in results_df['party'].unique():
        group_data = results_df[results_df['party'] == party]['degree_centrality'].values
        if len(group_data) >= 2:
            party_groups[party] = group_data

    if len(party_groups) >= 2:
        groups = list(party_groups.values())
        stat, p_value = stats.kruskal(*groups)
        print(f"  H-statistic: {stat:.4f}")
        print(f"  p-value:     {p_value:.6f}")
        if p_value < 0.05:
            print("  Result: REJECT H₀ – Significant difference found (α = 0.05)")
        else:
            print("  Result: FAIL TO REJECT H₀ – No significant difference (α = 0.05)")

        # ── 7e. Dunn's Post-Hoc Test ────────────────────────────────────────
        if p_value < 0.05 and HAS_POSTHOCS and len(party_groups) >= 3:
            print(f"\n  Dunn's Post-Hoc Test (Bonferroni correction):")
            print(f"  {'─' * 60}")
            dunn_result = sp.posthoc_dunn(
                results_df, val_col='degree_centrality', group_col='party',
                p_adjust='bonferroni'
            )
            print(dunn_result.round(4).to_string())

    # ── 7f. Chamber Comparison ───────────────────────────────────────────────
    print(f"\n  Degree Centrality by Chamber:")
    print(f"  {'─' * 60}")
    chamber_stats = results_df.groupby('chamber')['degree_centrality'].agg(
        ['count', 'mean', 'median', 'std']
    ).round(4)
    print(chamber_stats.to_string())

    # Mann-Whitney U test for Senate vs House
    senate = results_df[results_df['chamber'] == 'Senate']['degree_centrality'].values
    house = results_df[results_df['chamber'] == 'House of Representatives']['degree_centrality'].values
    if len(senate) > 0 and len(house) > 0:
        u_stat, u_p = stats.mannwhitneyu(senate, house, alternative='two-sided')
        print(f"\n  Mann-Whitney U Test (Senate vs House):")
        print(f"  U-statistic: {u_stat:.4f}")
        print(f"  p-value:     {u_p:.6f}")

    # ── 7g. Regional Comparison ──────────────────────────────────────────────
    print(f"\n  Degree Centrality by Region:")
    print(f"  {'─' * 60}")
    region_stats = results_df.groupby('region')['degree_centrality'].agg(
        ['count', 'mean', 'median', 'std']
    ).round(4)
    print(region_stats.to_string())

    print()
    return results_df


# =============================================================================
# VISUALIZATION
# =============================================================================
def create_visualizations(G, results_df, output_dir='.'):
    """
    Create publication-quality visualizations of the cosponsorship network.

    Parameters
    ----------
    G : nx.Graph
        The cosponsorship network.
    results_df : pd.DataFrame
        Centrality results with attributes.
    output_dir : str
        Directory to save figures.
    """
    print("=" * 70)
    print("VISUALIZATIONS")
    print("=" * 70)

    # Color map for parties
    party_colors = {
        'Democrat': '#2166AC',      # Blue
        'Republican': '#B2182B',    # Red
        'Independent': '#4DAF4A',   # Green
        'Unknown': '#999999'        # Gray
    }

    # ── Figure 1: Box Plot – Degree Centrality by Party ──────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    party_order = ['Democrat', 'Republican', 'Independent']
    available_parties = [p for p in party_order if p in results_df['party'].values]

    plot_data = results_df[results_df['party'].isin(available_parties)]

    sns.boxplot(
        data=plot_data, x='party', y='degree_centrality',
        order=available_parties,
        palette=[party_colors[p] for p in available_parties],
        width=0.5, ax=ax
    )
    sns.stripplot(
        data=plot_data, x='party', y='degree_centrality',
        order=available_parties,
        color='black', alpha=0.3, size=3, jitter=True, ax=ax
    )

    ax.set_title('Degree Centrality Distribution by Party Affiliation\n118th U.S. Congress',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Party', fontsize=12)
    ax.set_ylabel('Degree Centrality', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(f'{output_dir}/fig1_centrality_by_party.png', dpi=300, bbox_inches='tight')
    print("  Saved: fig1_centrality_by_party.png")
    plt.close()

    # ── Figure 2: Violin Plot – Centrality by Chamber ────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.violinplot(
        data=results_df, x='chamber', y='degree_centrality',
        palette=['#2166AC', '#B2182B'], inner='box', ax=ax
    )

    ax.set_title('Degree Centrality Distribution by Chamber\n118th U.S. Congress',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Chamber', fontsize=12)
    ax.set_ylabel('Degree Centrality', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(f'{output_dir}/fig2_centrality_by_chamber.png', dpi=300, bbox_inches='tight')
    print("  Saved: fig2_centrality_by_chamber.png")
    plt.close()

    # ── Figure 3: Network Visualization (Largest Component) ──────────────────
    fig, ax = plt.subplots(figsize=(14, 14))

    # Use the largest connected component for cleaner visualization
    if nx.is_connected(G):
        subgraph = G
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc).copy()

    # Sample if too large for clean visualization
    if subgraph.number_of_nodes() > 200:
        # Take top 200 by degree centrality
        dc = nx.degree_centrality(subgraph)
        top_nodes = sorted(dc, key=dc.get, reverse=True)[:200]
        subgraph = subgraph.subgraph(top_nodes).copy()

    # Layout
    pos = nx.spring_layout(subgraph, k=0.3, iterations=50, seed=42)

    # Node colors by party
    node_colors = [
        party_colors.get(subgraph.nodes[n].get('party', 'Unknown'), '#999999')
        for n in subgraph.nodes
    ]

    # Node sizes by degree centrality
    dc_sub = nx.degree_centrality(subgraph)
    node_sizes = [300 + 3000 * dc_sub[n] for n in subgraph.nodes]

    # Draw
    nx.draw_networkx_edges(subgraph, pos, alpha=0.05, width=0.5, ax=ax)
    nx.draw_networkx_nodes(
        subgraph, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.7,
        ax=ax
    )

    # Legend
    legend_patches = [
        mpatches.Patch(color=party_colors['Democrat'], label='Democrat'),
        mpatches.Patch(color=party_colors['Republican'], label='Republican'),
        mpatches.Patch(color=party_colors['Independent'], label='Independent'),
    ]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=11,
              framealpha=0.9, title='Party', title_fontsize=12)

    ax.set_title('Congressional Cosponsorship Network\n118th U.S. Congress (Top 200 by Centrality)',
                 fontsize=16, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    fig.savefig(f'{output_dir}/fig3_network_visualization.png', dpi=300, bbox_inches='tight')
    print("  Saved: fig3_network_visualization.png")
    plt.close()

    # ── Figure 4: Heatmap – Mean Centrality by Region × Party ────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    pivot = results_df.pivot_table(
        values='degree_centrality', index='region', columns='party',
        aggfunc='mean'
    )

    # Reorder columns
    col_order = [c for c in ['Democrat', 'Republican', 'Independent'] if c in pivot.columns]
    pivot = pivot[col_order]

    sns.heatmap(
        pivot, annot=True, fmt='.3f', cmap='RdYlBu_r',
        linewidths=0.5, ax=ax, cbar_kws={'label': 'Mean Degree Centrality'}
    )

    ax.set_title('Mean Degree Centrality by Region and Party\n118th U.S. Congress',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Party', fontsize=12)
    ax.set_ylabel('Region', fontsize=12)

    plt.tight_layout()
    fig.savefig(f'{output_dir}/fig4_heatmap_region_party.png', dpi=300, bbox_inches='tight')
    print("  Saved: fig4_heatmap_region_party.png")
    plt.close()

    print()


# =============================================================================
# DATA EXPORT
# =============================================================================
def export_data(members_df, edge_df, results_df, G, output_dir='.'):
    """
    Export all data artifacts for reproducibility and further analysis.
    """
    print("=" * 70)
    print("DATA EXPORT")
    print("=" * 70)

    # Save member data
    members_df.to_csv(f'{output_dir}/members_118th.csv', index=False)
    print(f"  Saved: members_118th.csv ({len(members_df)} rows)")

    # Save edge list
    edge_df.to_csv(f'{output_dir}/edge_list_118th.csv', index=False)
    print(f"  Saved: edge_list_118th.csv ({len(edge_df)} rows)")

    # Save centrality results
    results_df.to_csv(f'{output_dir}/centrality_results_118th.csv', index=False)
    print(f"  Saved: centrality_results_118th.csv ({len(results_df)} rows)")

    # Save NetworkX graph as GraphML
    nx.write_graphml(G, f'{output_dir}/cosponsorship_network_118th.graphml')
    print(f"  Saved: cosponsorship_network_118th.graphml")

    # Save summary statistics as JSON
    summary = {
        'congress': CONGRESS,
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'density': round(nx.density(G), 6),
        'is_connected': nx.is_connected(G),
        'party_mean_centrality': results_df.groupby('party')['degree_centrality'].mean().round(4).to_dict(),
        'chamber_mean_centrality': results_df.groupby('chamber')['degree_centrality'].mean().round(4).to_dict(),
    }
    with open(f'{output_dir}/network_summary_118th.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: network_summary_118th.json")

    print()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """
    Execute the full 7-step Congressional Cosponsorship Network Analysis pipeline.
    """
    print("\n" + "=" * 70)
    print("  CONGRESSIONAL COSPONSORSHIP NETWORK ANALYSIS")
    print("  118th U.S. Congress – All Bill Types")
    print("  Data Source: Congress.gov API (Library of Congress)")
    print("=" * 70 + "\n")

    # Create output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Environment Setup (handled by imports above) ─────────────────
    print("STEP 1: Environment Setup – Complete ✓\n")

    # ── Step 2: Retrieve Members ─────────────────────────────────────────────
    members_df = retrieve_members()

    # ── Step 3: Retrieve Bills and Cosponsors ────────────────────────────────
    # NOTE: For a full run with all bill types, this will make thousands of
    # API calls and may take several hours. Adjust max_bills_per_type for testing.
    #
    # For testing:  bill_data = retrieve_bill_cosponsors(max_bills_per_type=50)
    # For full run:  bill_data = retrieve_bill_cosponsors()
    bill_data = retrieve_bill_cosponsors(
        bill_types=['hr', 's', 'hjres', 'sjres', 'hconres', 'sconres', 'hres', 'sres'],
        max_bills_per_type=None  # Set to 50 or 100 for testing
    )

    # ── Step 4: Construct Edge List ──────────────────────────────────────────
    edge_df = construct_edge_list(bill_data)

    # ── Step 5: Build NetworkX Graph ─────────────────────────────────────────
    G = build_network(members_df, edge_df)

    # ── Step 6: Validate and Clean ───────────────────────────────────────────
    G = validate_and_clean(G, members_df)

    # ── Step 7: Compute Centrality and Analyze ───────────────────────────────
    results_df = compute_and_analyze_centrality(G)

    # ── Visualization ────────────────────────────────────────────────────────
    create_visualizations(G, results_df, output_dir)

    # ── Export ────────────────────────────────────────────────────────────────
    export_data(members_df, edge_df, results_df, G, output_dir)

    print("=" * 70)
    print("  ANALYSIS COMPLETE")
    print(f"  All outputs saved to: ./{output_dir}/")
    print("=" * 70)

    return G, members_df, edge_df, results_df


# =============================================================================
# QUICK-START MODE: Run with a small sample for testing
# =============================================================================
def quick_start():
    """
    Run with a small sample (50 bills per type) for quick testing.
    Use this to verify your API key works before a full run.
    """
    print("\n" + "=" * 70)
    print("  QUICK-START MODE (Sample of 50 bills per type)")
    print("  Use this to test your API key and verify the pipeline works.")
    print("=" * 70 + "\n")

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    print("STEP 1: Environment Setup – Complete ✓\n")

    members_df = retrieve_members()

    # Only Senate and House bills, limited to 50 each
    bill_data = retrieve_bill_cosponsors(
        bill_types=['hr', 's'],
        max_bills_per_type=50
    )

    edge_df = construct_edge_list(bill_data)
    G = build_network(members_df, edge_df)
    G = validate_and_clean(G, members_df)
    results_df = compute_and_analyze_centrality(G)
    create_visualizations(G, results_df, output_dir)
    export_data(members_df, edge_df, results_df, G, output_dir)

    print("=" * 70)
    print("  QUICK-START COMPLETE")
    print(f"  Outputs saved to: ./{output_dir}/")
    print("  Run main() for the full analysis with all bill types.")
    print("=" * 70)

    return G, members_df, edge_df, results_df


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # ┌─────────────────────────────────────────────────────────┐
    # │  Choose your run mode:                                  │
    # │                                                         │
    # │  quick_start()  → ~100 bills, runs in ~10-15 minutes    │
    # │  main()         → all bills, runs in several hours      │
    # │                                                         │
    # │  Recommendation: Start with quick_start() to verify     │
    # │  your API key works, then run main() for the full       │
    # │  analysis.                                              │
    # └─────────────────────────────────────────────────────────┘

    G, members_df, edge_df, results_df = quick_start()
    # G, members_df, edge_df, results_df = main()