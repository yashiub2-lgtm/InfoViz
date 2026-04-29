import json
import os
import glob
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import networkx as nx
from scipy.cluster.hierarchy import linkage, dendrogram


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REGION_COLORS = {
    "Americas": "#2E86AB",
    "Asia":     "#A23B72",
    "Europe":   "#F18F01",
    "Oceania":  "#3FB47C",
    "World":    "#8DB580",
}


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_sky_cultures(data_dir):
    """Read every *.json file in data_dir and return them as a list of dicts."""
    paths = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    cultures = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                cultures.append(json.load(f))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ! Skipped {os.path.basename(path)}: {exc}")
    return cultures


def extract_stars_from_lines(lines):
    """Flatten the list-of-lists 'lines' field into the set of unique stars used."""
    stars = set()
    for segment in lines or []:
        for star in segment:
            stars.add(star)
    return stars


def count_edges(lines):
    """A 'line' is a polyline; its number of edges is len(polyline) - 1."""
    edges = 0
    for segment in lines or []:
        if len(segment) >= 2:
            edges += len(segment) - 1
    return edges


def build_records(cultures):
    """Build per-culture and per-constellation tabular records."""
    culture_rows = []
    constellation_rows = []

    for culture in cultures:
        name = culture.get("name") or culture.get("id", "Unknown")
        region = culture.get("region", "Unknown")
        constellations = culture.get("constellations", []) or []

        culture_stars = set()
        culture_edges = 0
        lined_count = 0
        per_const_star_counts = []
        per_const_edge_counts = []

        for con in constellations:
            lines = con.get("lines", [])
            stars = extract_stars_from_lines(lines)
            edges = count_edges(lines)
            has_lines = bool(lines) and any(len(seg) >= 2 for seg in lines)

            if has_lines:
                lined_count += 1
            culture_stars |= stars
            culture_edges += edges
            per_const_star_counts.append(len(stars))
            per_const_edge_counts.append(edges)

            con_names = con.get("names", [{}])
            english = con_names[0].get("english") if con_names else None

            constellation_rows.append({
                "culture": name,
                "region": region,
                "constellation_id": con.get("id"),
                "english_name": english,
                "num_stars": len(stars),
                "num_edges": edges,
                "semantics": con.get("semantics", []) or [],
                "has_lines": has_lines,
            })

        culture_rows.append({
            "culture": name,
            "region": region,
            "subregion": culture.get("subregion"),
            "period": culture.get("period"),
            "num_constellations": len(constellations),
            "num_lined_constellations": lined_count,
            "unique_stars": len(culture_stars),
            "total_edges": culture_edges,
            "avg_stars_per_constellation": np.mean(per_const_star_counts) if per_const_star_counts else 0,
            "avg_edges_per_constellation": np.mean(per_const_edge_counts) if per_const_edge_counts else 0,
            "stars_set": culture_stars,
        })

    return pd.DataFrame(culture_rows), pd.DataFrame(constellation_rows)


# ---------------------------------------------------------------------------
# 2. Exploratory data analysis
# ---------------------------------------------------------------------------

def run_eda(culture_df, constellation_df):
    """Print summary statistics matching the report's Section 2.1 table."""
    print("=" * 70)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    total_cultures = len(culture_df)
    total_constellations = len(constellation_df)
    total_lined = int(constellation_df["has_lines"].sum())
    pct_lined = 100 * total_lined / total_constellations if total_constellations else 0

    all_stars = set()
    for s in culture_df["stars_set"]:
        all_stars |= s
    total_unique_stars = len(all_stars)

    total_edges = int(culture_df["total_edges"].sum())

    semantic_counter = Counter()
    for sem_list in constellation_df["semantics"]:
        for s in sem_list:
            semantic_counter[s] += 1

    print(f"\n  Total sky cultures                  : {total_cultures}")
    print(f"  Total constellations defined        : {total_constellations}")
    print(f"  Constellations with line figures    : {total_lined} ({pct_lined:.1f}%)")
    print(f"  Total unique stars referenced       : {total_unique_stars}")
    print(f"  Total constellation edges           : {total_edges}")
    print(f"  Unique semantic categories          : {len(semantic_counter)}")
    print(f"  Geographic regions                  : {culture_df['region'].nunique()}")

    print("\n  Cultures per region:")
    for region, n in culture_df["region"].value_counts().items():
        pct = 100 * n / total_cultures
        print(f"    {region:10s} {n:3d}  ({pct:.1f}%)")

    print("\n  Constellation count statistics (per culture):")
    desc = culture_df["num_constellations"].describe()
    for k, v in desc.items():
        print(f"    {k:8s} {v:.2f}")

    print("\n  Top 10 cultures by constellation count:")
    top10 = culture_df.nlargest(10, "num_constellations")[["culture", "region", "num_constellations"]]
    for _, row in top10.iterrows():
        print(f"    {row['culture']:20s} {row['region']:10s} {row['num_constellations']:3d}")

    print("\n  Semantic category frequencies:")
    for cat, count in semantic_counter.most_common():
        print(f"    {cat:18s} {count:3d}")

    print("\n  Stars per constellation summary:")
    desc_s = constellation_df["num_stars"].describe()
    for k, v in desc_s.items():
        print(f"    {k:8s} {v:.2f}")

    print("=" * 70 + "\n")
    return semantic_counter, all_stars


# ---------------------------------------------------------------------------
# 3. Network construction and metrics
# ---------------------------------------------------------------------------

def build_combined_graph(cultures):
    """Build a single graph whose edges are the union of all constellation edges."""
    G = nx.Graph()
    for culture in cultures:
        for con in culture.get("constellations", []) or []:
            for segment in con.get("lines", []) or []:
                for a, b in zip(segment[:-1], segment[1:]):
                    if a == b:
                        continue
                    G.add_edge(a, b)
    return G


def star_universality(culture_df):
    """For each star, count how many cultures use it."""
    star_count = Counter()
    for stars in culture_df["stars_set"]:
        for s in stars:
            star_count[s] += 1
    return star_count


def jaccard_matrix(culture_df):
    """Pairwise Jaccard similarity over the stars used by each culture."""
    names = culture_df["culture"].tolist()
    sets = culture_df["stars_set"].tolist()
    n = len(names)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i, j] = 1.0
            else:
                a, b = sets[i], sets[j]
                union = a | b
                M[i, j] = len(a & b) / len(union) if union else 0.0
    return pd.DataFrame(M, index=names, columns=names)


# ---------------------------------------------------------------------------
# 4. Visualizations
# ---------------------------------------------------------------------------

def fig2_constellations_per_culture(culture_df, out_path):
    """Horizontal bar chart of constellation counts, color-coded by region."""
    df = culture_df.sort_values("num_constellations", ascending=True)
    colors = [REGION_COLORS.get(r, "#888") for r in df["region"]]

    fig, ax = plt.subplots(figsize=(9, 12))
    ax.barh(df["culture"], df["num_constellations"], color=colors, edgecolor="white")
    ax.set_xlabel("Number of Constellations")
    ax.set_title("Constellations per Sky Culture", fontsize=14, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)

    handles = [mpatches.Patch(color=c, label=r) for r, c in REGION_COLORS.items()
               if r in culture_df["region"].unique()]
    ax.legend(handles=handles, loc="lower right", title=None)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig3_region_pie(culture_df, out_path):
    """Pie chart of cultures per geographic region."""
    counts = culture_df["region"].value_counts()
    colors = [REGION_COLORS.get(r, "#888") for r in counts.index]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.0f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 11},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax.set_title("Sky Cultures by Geographic Region", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig4_semantic_categories(semantic_counter, out_path):
    """Horizontal bar chart of semantic category frequencies."""
    items = semantic_counter.most_common()
    labels = [k for k, _ in items][::-1]
    values = [v for _, v in items][::-1]

    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(values) - 1)) for i in range(len(values))]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(labels, values, color=colors, edgecolor="white")
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(value), va="center", fontsize=9)

    ax.set_xlabel("Frequency Across All Cultures")
    ax.set_title("Semantic Categories of Constellations", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig5_star_universality(star_count, out_path):
    """Histogram showing how many cultures share each star."""
    counts = list(star_count.values())
    top_stars = star_count.most_common(7)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(counts, bins=range(1, max(counts) + 2), color="#3E5C76",
            edgecolor="white")
    ax.set_xlabel("Number of Cultures Sharing a Star")
    ax.set_ylabel("Number of Stars")
    ax.set_title("Star Universality Distribution", fontsize=14, fontweight="bold")

    legend_text = "Most Universal Stars:\n" + "\n".join(
        f"  {name}: {n} cultures" for name, n in top_stars
    )
    ax.text(0.98, 0.98, legend_text, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(facecolor="#E8F0F7", edgecolor="#9BB7D4",
                                  boxstyle="round,pad=0.4"))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig6_dendrogram(culture_df, jacc, out_path):
    """Hierarchical clustering dendrogram of cultures based on Jaccard distance."""
    names = jacc.index.tolist()
    distance = 1.0 - jacc.values
    np.fill_diagonal(distance, 0.0)
    # Convert square distance matrix to condensed form for linkage
    iu = np.triu_indices(len(names), k=1)
    condensed = distance[iu]
    Z = linkage(condensed, method="average")

    region_lookup = dict(zip(culture_df["culture"], culture_df["region"]))

    fig, ax = plt.subplots(figsize=(14, 7))
    ddata = dendrogram(Z, labels=names, leaf_rotation=90, leaf_font_size=9, ax=ax)

    # Color leaf labels by region
    for tick in ax.get_xticklabels():
        region = region_lookup.get(tick.get_text(), "Unknown")
        tick.set_color(REGION_COLORS.get(region, "#333"))

    ax.set_ylabel("Distance (1 - Jaccard Similarity)")
    ax.set_title("Hierarchical Clustering of Sky Cultures by Shared Stars",
                 fontsize=14, fontweight="bold")

    handles = [mpatches.Patch(color=c, label=r) for r, c in REGION_COLORS.items()
               if r in culture_df["region"].unique()]
    ax.legend(handles=handles, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig7_complexity_bubble(culture_df, out_path):
    """Scatter of constellation count vs. avg edges, bubble area = avg stars per constellation."""
    fig, ax = plt.subplots(figsize=(11, 7))

    for region, color in REGION_COLORS.items():
        sub = culture_df[culture_df["region"] == region]
        if sub.empty:
            continue
        ax.scatter(
            sub["num_constellations"],
            sub["avg_edges_per_constellation"],
            s=sub["avg_stars_per_constellation"] * 25,
            color=color,
            alpha=0.55,
            edgecolor="black",
            linewidth=0.6,
            label=region,
        )

    # Label notable cultures (top-N by avg edges, plus the largest catalogs)
    notable = pd.concat([
        culture_df.nlargest(6, "avg_edges_per_constellation"),
        culture_df.nlargest(4, "num_constellations"),
    ]).drop_duplicates("culture")
    for _, row in notable.iterrows():
        ax.annotate(row["culture"],
                    (row["num_constellations"], row["avg_edges_per_constellation"]),
                    fontsize=8, xytext=(5, 4), textcoords="offset points")

    ax.set_xlabel("Number of Constellations")
    ax.set_ylabel("Avg. Edges per Constellation")
    ax.set_title("Constellation Complexity by Culture", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig8_degree_distribution(graph, out_path):
    """Bar chart of node-degree distribution from the combined constellation network."""
    degrees = [d for _, d in graph.degree()]
    deg_counts = Counter(degrees)

    top_hubs = sorted(graph.degree(), key=lambda x: x[1], reverse=True)[:7]

    xs = sorted(deg_counts.keys())
    ys = [deg_counts[x] for x in xs]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(xs, ys, color="#3E5C76", edgecolor="white", width=0.7)
    ax.set_xlabel("Degree (number of connections)")
    ax.set_ylabel("Number of Stars")
    ax.set_title("Star Degree Distribution Across All Cultures",
                 fontsize=14, fontweight="bold")

    legend_text = "Top Hub Stars:\n" + "\n".join(
        f"  {name}: degree {d}" for name, d in top_hubs
    )
    ax.text(0.98, 0.98, legend_text, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(facecolor="#FAF3E0", edgecolor="#C5A572",
                                  boxstyle="round,pad=0.4"))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig9_jaccard_heatmap(culture_df, jacc, out_path, top_n=20):
    """Heatmap of Jaccard similarity for the top-N cultures by constellation count."""
    top = culture_df.nlargest(top_n, "num_constellations")["culture"].tolist()
    sub = jacc.loc[top, top]

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(sub, cmap="YlOrRd", vmin=0, vmax=0.7, square=True,
                cbar_kws={"label": "Jaccard Similarity"}, ax=ax)
    ax.set_title(f"Star-Sharing Similarity Between Top {top_n} Cultures",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig10_semantic_radar(culture_df, constellation_df, out_path, top_n=5):
    """Radar chart of semantic profiles for the largest sky cultures."""
    top = culture_df.nlargest(top_n, "num_constellations")["culture"].tolist()

    categories = ["bird", "humanoid", "mammal", "fish", "reptile",
                  "geometric", "landscape", "man-made object"]

    profiles = {}
    for culture in top:
        sub = constellation_df[constellation_df["culture"] == culture]
        total_labels = sum(len(s) for s in sub["semantics"]) or 1
        counts = Counter()
        for sem_list in sub["semantics"]:
            for s in sem_list:
                counts[s] += 1
        profiles[culture] = [counts.get(c, 0) / total_labels for c in categories]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    palette = ["#E63946", "#1D8FE6", "#2A9D8F", "#F4A261", "#9D4EDD"]
    for (culture, values), color in zip(profiles.items(), palette):
        values_closed = values + values[:1]
        ax.plot(angles, values_closed, color=color, linewidth=2, label=culture)
        ax.fill(angles, values_closed, color=color, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title(f"Semantic Profiles of Top {top_n} Cultures",
                 fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading sky-culture JSON files from {DATA_DIR} ...")
    cultures = load_sky_cultures(DATA_DIR)
    print(f"Loaded {len(cultures)} sky-culture files.\n")

    print("Building tabular records ...")
    culture_df, constellation_df = build_records(cultures)

    semantic_counter, all_stars = run_eda(culture_df, constellation_df)

    print("Building combined constellation network ...")
    G = build_combined_graph(cultures)
    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    print("Computing star-universality counts ...")
    star_count = star_universality(culture_df)

    print("Computing pairwise Jaccard similarity ...")
    jacc = jaccard_matrix(culture_df)

    # Save tabular outputs alongside the figures
    culture_df.drop(columns=["stars_set"]).to_csv(
        os.path.join(OUTPUT_DIR, "culture_summary.csv"), index=False
    )
    constellation_df.to_csv(
        os.path.join(OUTPUT_DIR, "constellation_summary.csv"), index=False
    )

    print("Generating figures ...")
    fig2_constellations_per_culture(culture_df, os.path.join(OUTPUT_DIR, "Fig_2.png"))
    print("  Fig 2: constellations per culture")
    fig3_region_pie(culture_df, os.path.join(OUTPUT_DIR, "Fig_3.png"))
    print("  Fig 3: region pie chart")
    fig4_semantic_categories(semantic_counter, os.path.join(OUTPUT_DIR, "Fig_4.png"))
    print("  Fig 4: semantic category frequencies")
    fig5_star_universality(star_count, os.path.join(OUTPUT_DIR, "Fig_5.png"))
    print("  Fig 5: star universality histogram")
    fig6_dendrogram(culture_df, jacc, os.path.join(OUTPUT_DIR, "Fig_6.png"))
    print("  Fig 6: hierarchical clustering dendrogram")
    fig7_complexity_bubble(culture_df, os.path.join(OUTPUT_DIR, "Fig_7.png"))
    print("  Fig 7: constellation complexity bubble chart")
    fig8_degree_distribution(G, os.path.join(OUTPUT_DIR, "Fig_8.png"))
    print("  Fig 8: star degree distribution")
    fig9_jaccard_heatmap(culture_df, jacc, os.path.join(OUTPUT_DIR, "Fig_9.png"))
    print("  Fig 9: Jaccard similarity heatmap (top 20)")
    fig10_semantic_radar(culture_df, constellation_df, os.path.join(OUTPUT_DIR, "Fig_10.png"))
    print("  Fig 10: semantic profile radar chart")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
