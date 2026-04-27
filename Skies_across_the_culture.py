import argparse
import json
import os
import re
from collections import Counter, defaultdict
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Region palette is fixed across every figure so a reader who has learned
# the encoding in Fig. 3 can carry it through the rest of the report.
REGION_COLORS = {
    "Americas": "#2E86AB",
    "Asia":     "#A23B72",
    "Europe":   "#E68A1E",
    "Oceania":  "#3CB371",
    "World":    "#9DC183",
}

# Region order also drives the pie chart so the slice arrangement matches
# the report exactly.
REGION_ORDER = ["Americas", "Asia", "Europe", "World", "Oceania"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Data loading & helpers
# ---------------------------------------------------------------------------

def _normalize_star(star_id):
    """Normalize a SIMBAD-style star id.

    Per Section 2.2 of the report: "star names sometimes use different
    SIMBAD-style notations, so we preserved original labels while grouping
    identical references where possible." Normalization here is conservative:
    leading "* " marker is stripped and internal whitespace is collapsed.
    Case is preserved so Bayer designations (alf vs. Alf) are not silently
    merged unless they truly are identical.
    """
    if star_id is None:
        return ""
    cleaned = star_id.strip()
    if cleaned.startswith("* "):
        cleaned = cleaned[2:].strip()
    return re.sub(r"\s+", " ", cleaned)


def _has_usable_ids(polyline):
    """Per Section 2.2: ~3% of line figures lack usable star IDs."""
    return all(s and s.strip() for s in polyline)


def load_cultures(data_dir):
    """Load every *.json sky-culture file in data_dir."""
    cultures = []
    for path in sorted(glob(os.path.join(data_dir, "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except (json.JSONDecodeError, OSError):
            continue
        if "constellations" in data and "region" in data:
            data["_filename"] = os.path.basename(path)
            cultures.append(data)
    return cultures


def lined_constellations(culture):
    """Constellations that contain at least one usable line figure."""
    return [c for c in culture.get("constellations", [])
            if c.get("lines") and any(_has_usable_ids(p) and len(p) >= 2
                                      for p in c["lines"])]


def constellation_edges(constellation):
    """Convert polyline lists into a set of undirected, normalized edges."""
    edges = set()
    for poly in constellation.get("lines", []):
        if not _has_usable_ids(poly):
            continue
        norm = [_normalize_star(s) for s in poly]
        for a, b in zip(norm, norm[1:]):
            if a != b and a and b:
                edges.add(tuple(sorted((a, b))))
    return edges


def constellation_segment_count(constellation):
    """Total line segments (with multiplicity), as counted in Section 2.2."""
    n = 0
    for poly in constellation.get("lines", []):
        if not _has_usable_ids(poly):
            continue
        n += max(0, len(poly) - 1)
    return n


def culture_stars(culture):
    """All normalized star ids that appear in any line figure of a culture."""
    stars = set()
    for c in lined_constellations(culture):
        for poly in c["lines"]:
            if _has_usable_ids(poly):
                stars.update(_normalize_star(s) for s in poly)
    stars.discard("")
    return stars


def culture_edges(culture):
    """Undirected (star, star) edges, deduplicated within the culture."""
    edges = set()
    for c in lined_constellations(culture):
        edges |= constellation_edges(c)
    return edges


def print_dataset_summary(cultures):
    """Match Section 2.2 of the report so reproducibility can be audited."""
    n_cultures = len(cultures)
    n_total_const = sum(len(c.get("constellations", [])) for c in cultures)
    n_lined_const = sum(len(lined_constellations(c)) for c in cultures)

    all_stars = set()
    total_segments = 0
    for c in cultures:
        all_stars |= culture_stars(c)
        for con in lined_constellations(c):
            total_segments += constellation_segment_count(con)

    sem_categories = set()
    for c in cultures:
        for con in lined_constellations(c):
            for tag in con.get("semantics") or []:
                if tag:
                    sem_categories.add(tag.strip().lower())

    regions = Counter(c["region"] for c in cultures)

    print("Dataset summary (compare to report §2.2):")
    print(f"  sky cultures ............... {n_cultures}")
    print(f"  named constellations ....... {n_total_const}")
    print(f"  with line figures .......... {n_lined_const}")
    print(f"  unique referenced stars .... {len(all_stars)}")
    print(f"  star-to-star line segments . {total_segments}")
    print(f"  semantic categories ........ {len(sem_categories)}")
    print(f"  region labels .............. {len(regions)}")
    for r in REGION_ORDER:
        if r in regions:
            print(f"    {r:10s} {regions[r]:3d}  "
                  f"({100*regions[r]/n_cultures:.1f}%)")


def _save_caption(out_path, caption):
    """Sidecar caption file matching the figure name (.png -> .txt).

    The report says "captions identify the variables, the comparison task,
    and the main takeaway." Storing them as sidecar files lets the GitHub
    appendix host figure + caption pairs without losing the wording in
    matplotlib title overflow.
    """
    txt_path = os.path.splitext(out_path)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as fp:
        fp.write(caption.strip() + "\n")


# ---------------------------------------------------------------------------
# 1. Constellations per Sky Culture
# ---------------------------------------------------------------------------

def plot_constellations_per_culture(cultures, out_path):
    rows = [(c["name"], len(lined_constellations(c)), c["region"])
            for c in cultures]
    rows = [r for r in rows if r[1] > 0]
    rows.sort(key=lambda r: r[1])

    names  = [r[0] for r in rows]
    counts = [r[1] for r in rows]
    colors = [REGION_COLORS.get(r[2], "#888888") for r in rows]

    fig, ax = plt.subplots(figsize=(9, max(8, 0.22 * len(rows))))
    ax.barh(names, counts, color=colors, edgecolor="white")
    ax.set_xlabel("Number of Constellations")
    ax.set_title("Constellations per Sky Culture", fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)

    legend_handles = [Patch(facecolor=col, label=reg)
                      for reg, col in REGION_COLORS.items()]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    _save_caption(out_path,
        "Fig. A1 (appendix). Constellation catalog size per sky culture, "
        "colored by region. Task: show how unevenly distributed catalog "
        "size is across the 40-culture sample. Takeaway: a small group "
        "(Dien, IAU, Ruelle, Romania) defines an order of magnitude more "
        "constellations than most cultures, so raw counts cannot be "
        "compared without context. Limitation: catalog size reflects "
        "documentation depth, not cultural importance — many small "
        "catalogs (e.g. several Dene cultures) cover only a few "
        "well-attested figures.")


# ---------------------------------------------------------------------------
# 2. Sky Cultures by Geographic Region (REPORT FIG 3)
# ---------------------------------------------------------------------------

def plot_region_pie(cultures, out_path):
    counter = Counter(c["region"] for c in cultures)
    labels = [r for r in REGION_ORDER if r in counter] + \
             [r for r in counter if r not in REGION_ORDER]
    sizes  = [counter[r] for r in labels]
    colors = [REGION_COLORS.get(r, "#888888") for r in labels]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.0f%%",
        startangle=90, counterclock=False,
        textprops={"fontsize": 12, "fontweight": "bold"},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    ax.set_title("Sky Cultures by Geographic Region",
                 fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    _save_caption(out_path,
        "Fig. 3 (report). Distribution of sky cultures by geographic "
        "region. Task: communicate dataset coverage and imbalance "
        "before any cross-cultural comparison is interpreted. Takeaway: "
        "the Americas (~58%) and Asia (~30%) dominate the sample; "
        "Europe, Oceania, and World contribute only 5 cultures combined. "
        "Limitation: any cross-regional similarity, clustering, or "
        "semantic claim must be read as a pattern in this collection, "
        "not as a universal claim about world sky traditions.")


# ---------------------------------------------------------------------------
# 3. Semantic Categories of Constellations
# ---------------------------------------------------------------------------

def plot_semantic_categories(cultures, out_path):
    counter = Counter()
    for cult in cultures:
        for con in lined_constellations(cult):
            for tag in con.get("semantics") or []:
                if tag:
                    counter[tag.strip()] += 1

    if not counter:
        return

    items  = counter.most_common()
    labels = [k for k, _ in items][::-1]
    counts = [v for _, v in items][::-1]

    cmap   = plt.cm.viridis
    colors = cmap(np.linspace(0.15, 0.85, len(labels)))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(labels, counts, color=colors, edgecolor="white")
    ax.set_xlabel("Frequency Across All Cultures")
    ax.set_title("Semantic Categories of Constellations", fontweight="bold")

    for bar, val in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    _save_caption(out_path,
        "Fig. A2 (appendix). Frequency of the 15 normalized semantic "
        "categories across all line-figured constellations. Task: show "
        "what kinds of objects are projected into the sky. Takeaway: "
        "man-made objects, mammals, and humanoids dominate; abstract, "
        "plant, and architecture labels are rare. Limitation: many "
        "constellations carry more than one tag, so totals exceed the "
        "constellation count and the chart describes label frequency, "
        "not mutually exclusive shares.")


# ---------------------------------------------------------------------------
# 4. Star Universality Distribution
# ---------------------------------------------------------------------------

def plot_star_universality(cultures, out_path):
    star_to_cultures = defaultdict(set)
    for cult in cultures:
        name = cult["name"]
        for s in culture_stars(cult):
            star_to_cultures[s].add(name)

    if not star_to_cultures:
        return

    counts = np.array([len(v) for v in star_to_cultures.values()])
    top_stars = sorted(star_to_cultures.items(),
                       key=lambda kv: len(kv[1]), reverse=True)[:6]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(counts, bins=range(1, counts.max() + 2),
            color="#3D6B8C", edgecolor="white")
    ax.set_xlabel("Number of Cultures Sharing a Star")
    ax.set_ylabel("Number of Stars")
    ax.set_title("Star Universality Distribution", fontweight="bold")

    box_lines = ["Most Universal Stars:"]
    for star, cult_set in top_stars:
        box_lines.append(f"{star}: {len(cult_set)} cultures")
    ax.text(0.98, 0.98, "\n".join(box_lines),
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor="#888888"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    _save_caption(out_path,
        "Fig. A3 (appendix). Number of cultures sharing each referenced "
        "star. Task: identify anchor stars that recur across traditions. "
        "Takeaway: most stars appear in only one culture — universality "
        "is rare — and a small set of bright Big Dipper stars "
        "(delta/eta/zeta/gamma/epsilon/beta UMa) anchors the long tail, "
        "appearing in roughly 30 cultures. Limitation: appearance counts "
        "depend on which stars were positively identified by each "
        "source; cultures with thinner documentation contribute fewer "
        "counts even if their real sky knowledge was broader.")


# ---------------------------------------------------------------------------
# 5. Constellation Complexity by Culture
# ---------------------------------------------------------------------------

def plot_constellation_complexity(cultures, out_path):
    rows = []
    for cult in cultures:
        cons = lined_constellations(cult)
        if not cons:
            continue
        n_const = len(cons)
        # The report frames complexity as average edges per constellation;
        # we count line segments per constellation so a single long polyline
        # shows up as the high-complexity figure it is.
        seg_counts = [constellation_segment_count(c) for c in cons]
        avg_edges = float(np.mean(seg_counts)) if seg_counts else 0.0
        n_stars = len(culture_stars(cult))
        rows.append((cult["name"], n_const, avg_edges, n_stars, cult["region"]))

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(9, 6.5))

    max_stars = max(r[3] for r in rows) or 1
    for name, n_const, avg_edges, n_stars, region in rows:
        size = 30 + (n_stars / max_stars) * 800
        ax.scatter(n_const, avg_edges,
                   s=size, color=REGION_COLORS.get(region, "#888888"),
                   alpha=0.7, edgecolors="black", linewidth=0.5)

    # Label the cultures at the extremes (high avg-edges or high catalog size).
    by_complexity = sorted(rows, key=lambda r: r[2], reverse=True)[:6]
    by_size       = sorted(rows, key=lambda r: r[1], reverse=True)[:4]
    label_set = {r[0] for r in by_complexity} | {r[0] for r in by_size}
    for name, n_const, avg_edges, _, _ in rows:
        if name in label_set:
            ax.annotate(name, (n_const, avg_edges),
                        xytext=(5, 0), textcoords="offset points",
                        fontsize=8)

    ax.set_xlabel("Number of Constellations")
    ax.set_ylabel("Avg. Edges per Constellation")
    ax.set_title("Constellation Complexity by Culture", fontweight="bold")

    legend_handles = [Patch(facecolor=col, label=reg)
                      for reg, col in REGION_COLORS.items()]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    _save_caption(out_path,
        "Fig. A4 (appendix). Catalog size vs. average line-edges per "
        "constellation; bubble area is the total number of stars used "
        "by the culture. Task: show that catalog size and per-figure "
        "complexity are independent axes. Takeaway: a few small "
        "catalogs (e.g. Lower Tanana, Sahtuotine, Ahtna, Gwich'in) "
        "contain a single very long, highly connected figure, so their "
        "average is high; the largest catalogs (Dien, IAU, Ruelle) "
        "average only ~10 edges per constellation. Limitation: an "
        "average over a small catalog is high-variance — Lower Tanana "
        "has only one figure, so the y-axis value is that single "
        "constellation, not a stable summary of the culture.")


# ---------------------------------------------------------------------------
# 6. Hierarchical Clustering of Sky Cultures by Shared Stars (REPORT FIG 6)
# ---------------------------------------------------------------------------

def _jaccard(a, b):
    if not a and not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def plot_hierarchical_clustering(cultures, out_path):
    cults = [c for c in cultures if culture_stars(c)]
    names      = [c["name"] for c in cults]
    regions    = [c["region"] for c in cults]
    star_sets  = [culture_stars(c) for c in cults]

    n = len(cults)
    if n < 3:
        return

    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = 1.0 - _jaccard(star_sets[i], star_sets[j])
            dist[i, j] = dist[j, i] = d

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")

    fig, ax = plt.subplots(figsize=(11, 6))
    dendrogram(
        Z, labels=names, leaf_rotation=90, leaf_font_size=8,
        color_threshold=0.7, above_threshold_color="#999999",
        ax=ax,
    )

    name_to_region = dict(zip(names, regions))
    for lbl in ax.get_xmajorticklabels():
        lbl.set_color(REGION_COLORS.get(name_to_region.get(lbl.get_text(), ""),
                                        "#333333"))

    ax.set_ylabel("Distance (1 - Jaccard Similarity)")
    ax.set_title("Hierarchical Clustering of Sky Cultures by Shared Stars",
                 fontweight="bold")

    legend_handles = [Patch(facecolor=col, label=reg)
                      for reg, col in REGION_COLORS.items()]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    _save_caption(out_path,
        "Fig. 6 (report). Average-linkage hierarchical clustering of "
        "sky cultures using 1 - Jaccard similarity over shared star "
        "sets. Task: answer the project's central comparison question "
        "- which traditions use the most overlapping stars? Takeaway: "
        "Dien, IAU, and Ruelle cluster tightly - the three large "
        "European-/world-scope catalogs share most of their stars; "
        "Americas and Asia cultures form smaller, more dispersed "
        "groupings, so geography alone does not predict every cluster. "
        "Limitation: the metric measures shared stars, not shared "
        "*figures*; two cultures can use the same bright stars while "
        "drawing very different constellations.")


# ---------------------------------------------------------------------------
# 7. Star Degree Distribution Across All Cultures
# ---------------------------------------------------------------------------

def plot_star_degree_distribution(cultures, out_path):
    star_neighbors = defaultdict(set)
    for cult in cultures:
        for a, b in culture_edges(cult):
            star_neighbors[a].add(b)
            star_neighbors[b].add(a)

    if not star_neighbors:
        return

    degrees = np.array([len(v) for v in star_neighbors.values()])
    top_stars = sorted(star_neighbors.items(),
                       key=lambda kv: len(kv[1]), reverse=True)[:7]

    bins = np.arange(1, degrees.max() + 2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(degrees, bins=bins, color="#3D6B8C", edgecolor="white", rwidth=0.85)
    ax.set_xlabel("Degree (number of connections)")
    ax.set_ylabel("Number of Stars")
    ax.set_title("Star Degree Distribution Across All Cultures",
                 fontweight="bold")

    box_lines = ["Top Hub Stars:"]
    for star, neighbors in top_stars:
        box_lines.append(f"{star}: degree {len(neighbors)}")
    ax.text(0.98, 0.98, "\n".join(box_lines),
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor="#888888"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    _save_caption(out_path,
        "Fig. A5 (appendix). Degree distribution of the union network "
        "(every star-to-star line segment from every culture, "
        "deduplicated within each culture, then unioned). Task: show "
        "the structural shape of the combined network. Takeaway: the "
        "distribution is heavy-tailed - most stars connect to only "
        "two or three others, and a small set of bright hub stars "
        "(beta UMa, beta Aur, alpha Gem, alpha CMa, beta Ori, gamma "
        "Ori, delta UMa) accumulate ten or more connections across "
        "cultures. Limitation: this is an aggregate network and the "
        "high-degree hubs largely reflect how often the *same* bright "
        "stars are reused, not within-culture connectivity.")


# ---------------------------------------------------------------------------
# 8. Star-Sharing Similarity Between Top 20 Cultures (heatmap)
# ---------------------------------------------------------------------------

def plot_similarity_heatmap(cultures, out_path, top_n=20):
    ranked = sorted(cultures,
                    key=lambda c: len(lined_constellations(c)),
                    reverse=True)
    top = [c for c in ranked if culture_stars(c)][:top_n]
    if len(top) < 2:
        return

    names = [c["name"] for c in top]
    star_sets = [culture_stars(c) for c in top]
    n = len(top)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = _jaccard(star_sets[i], star_sets[j]) if i != j else 1.0

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(sim, cmap="YlOrRd", vmin=0, vmax=max(0.7, sim.max()))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title(f"Star-Sharing Similarity Between Top {n} Cultures",
                 fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Jaccard Similarity")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    _save_caption(out_path,
        "Fig. A6 (appendix). Pairwise Jaccard similarity over star sets "
        "for the 20 cultures with the largest line-figured catalogs. "
        "Task: focused, sortable cross-cultural comparison without the "
        "scale problems of an all-cultures heatmap. Takeaway: the "
        "Dien / IAU / Ruelle block stands out as the only region of "
        "high similarity (~0.5-0.7); most other pairs share <0.2 of "
        "their stars. Limitation: the report notes that heatmaps "
        "become crowded beyond ~25 cultures, so this view is "
        "deliberately truncated to the top 20 by catalog size.")


# ---------------------------------------------------------------------------
# 9. Semantic Profiles of Top 5 Cultures (radar)
# ---------------------------------------------------------------------------

def plot_semantic_radar(cultures, out_path, top_n=5):
    ranked = sorted(cultures,
                    key=lambda c: len(lined_constellations(c)),
                    reverse=True)[:top_n]
    if not ranked:
        return

    global_counter = Counter()
    per_culture = []
    for cult in ranked:
        local = Counter()
        total = 0
        for con in lined_constellations(cult):
            for tag in con.get("semantics") or []:
                if not tag:
                    continue
                key = tag.strip()
                local[key] += 1
                global_counter[key] += 1
                total += 1
        per_culture.append((cult["name"], local, total))

    categories = [t for t, _ in global_counter.most_common(8)]
    if not categories:
        return

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 7.5), subplot_kw={"polar": True})

    palette = ["#E63946", "#1D7AA2", "#2A9D8F", "#E9A23B", "#7B4B94",
               "#264653", "#F4A261"]

    for idx, (name, counts, total) in enumerate(per_culture):
        if total == 0:
            continue
        values = [counts.get(cat, 0) / total for cat in categories]
        values += values[:1]
        color = palette[idx % len(palette)]
        ax.plot(angles, values, color=color, linewidth=2, label=name)
        ax.fill(angles, values, color=color, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title(f"Semantic Profiles of Top {top_n} Cultures",
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    _save_caption(out_path,
        "Fig. A7 (appendix). Semantic fingerprints for the five "
        "largest catalogs, computed as the share of each culture's "
        "constellations falling into the eight most-common semantic "
        "labels. Task: compare how cultures distribute their "
        "constellation themes. Takeaway: even cultures with similar "
        "star sets (Dien, IAU, Ruelle) emphasize different label "
        "mixes - IAU leans humanoid, Bororo leans reptile, Romania "
        "leans man-made object. Limitation: radar charts become "
        "unreadable when too many profiles overlap, so this view is "
        "capped at five cultures and interactive explorations are "
        "deferred to the GitHub appendix.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="/mnt/project",
                        help="Directory containing sky-culture JSON files.")
    parser.add_argument("--out-dir", default="./output",
                        help="Directory to write the PNG visualizations "
                             "and sidecar caption files.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cultures = load_cultures(args.data_dir)
    print(f"Loaded {len(cultures)} sky cultures from {args.data_dir}\n")
    print_dataset_summary(cultures)
    print()

    plots = [
        ("01_constellations_per_culture.png",  plot_constellations_per_culture),
        ("02_region_pie.png",                  plot_region_pie),
        ("03_semantic_categories.png",         plot_semantic_categories),
        ("04_star_universality.png",           plot_star_universality),
        ("05_constellation_complexity.png",    plot_constellation_complexity),
        ("06_hierarchical_clustering.png",     plot_hierarchical_clustering),
        ("07_star_degree_distribution.png",    plot_star_degree_distribution),
        ("08_similarity_heatmap.png",          plot_similarity_heatmap),
        ("09_semantic_radar.png",              plot_semantic_radar),
    ]

    print("Generating figures:")
    for fname, fn in plots:
        out_path = os.path.join(args.out_dir, fname)
        try:
            fn(cultures, out_path)
            print(f"  wrote {out_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {fname}: {exc}")


if __name__ == "__main__":
    main()
