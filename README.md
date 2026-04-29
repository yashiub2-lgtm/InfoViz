# 🌌 Skies Across Cultures

A data-driven exploration of how 40 different cultures organize the night sky into constellations, treating constellation line figures as networks to make cross-cultural similarities and differences measurable.

## 📊 Overview

This project analyzes constellation data across 40 sky cultures to uncover:

- Shared stars across cultures
- Structural complexity of constellations
- Semantic themes (animals, objects, humanoids, etc.)
- Cross-cultural similarities through hierarchical clustering and Jaccard similarity

## 👥 Team & Client

**Team:** Carlos Banks, Dhruv Jore, Priyanshu Laddha, Yashkumar Burnwal  
**Client:** Prof. Doina Bucur

## 📂 Project Structure
InfoViz/
├── skies_across_cultures.py   # Main analysis script
├── README.md
├── requirements.txt
├── *.json                     # 40 sky-culture data files (Dien.json, IAU.json, ...)
├── Fig_2.png  ...  Fig_10.png
├── culture_summary.csv
└── constellation_summary.csv

## 🔧 Setup

### 1. Clone the repository

```bash
git clone https://github.com/yashiub2-lgtm/InfoViz.git
cd InfoViz
```

### 2. Create a virtual environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**Windows (cmd):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install directly:

```bash
pip install numpy pandas matplotlib seaborn networkx scipy
```

## ▶️ Running the Analysis

From the project root (with the virtual environment activated):

```bash
python skies_across_cultures.py
```

The script will:

1. Load all 40 `*.json` sky-culture files
2. Print exploratory data analysis to the console (counts, distributions, top cultures, semantic frequencies, network statistics)
3. Generate 9 figures (`Fig_2.png` through `Fig_10.png`) in the `outputs/` folder
4. Export `culture_summary.csv` and `constellation_summary.csv` for downstream analysis

Expected runtime: ~10–20 seconds on a modern laptop.

## 📈 Generated Visualizations

| Figure | Description |
| --- | --- |
| Fig 2 | Constellations per sky culture (horizontal bar, color-coded by region) |
| Fig 3 | Geographic distribution of cultures (pie chart) |
| Fig 4 | Frequency of semantic categories across all constellations |
| Fig 5 | Star universality histogram — how many cultures share each star |
| Fig 6 | Hierarchical clustering dendrogram based on Jaccard distance |
| Fig 7 | Constellation complexity bubble chart (count vs. avg edges, sized by avg stars) |
| Fig 8 | Star degree distribution from the combined network |
| Fig 9 | Jaccard similarity heatmap for the top 20 cultures |
| Fig 10 | Semantic profile radar chart for the top 5 cultures |

## 🔍 Key Insights

- **Heavy regional imbalance**: Americas account for 57.5% of cultures, Europe just 7.5% — but European catalogs (Dien, IAU, Ruelle) are far larger.
- **Most stars are culture-specific**: ~310 stars appear in only one culture, while only a handful (delta UMa, eta UMa, zeta UMa) appear in 30+ cultures.
- **Small catalogs can be complex**: Lower Tanana defines only 2 constellations but averages ~67 stars each, vs. IAU's ~9 stars per constellation across 86 entries.
- **Scale-free network structure**: Star degree distribution follows a power-law-like pattern, suggesting common structural principles across traditions.
- **Cultural fingerprints**: Romania emphasizes humanoids, Gond emphasizes man-made objects, while IAU shows a balanced semantic profile.

## 🛠️ Tech Stack

- **Python 3.8+**
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — static visualizations
- **networkx** — graph construction and degree analysis
- **scipy** — hierarchical clustering (linkage, dendrogram)

## 📚 Dataset

The constellation dataset was provided by the project client. It consists of 40 JSON files, each describing one sky culture's constellations as polylines connecting named stars, along with semantic labels and regional metadata.

**Dataset summary:**
- 40 sky cultures across 5 regions
- 539 constellations (97.0% with line figures)
- 1,068 unique stars referenced
- 4,363 constellation edges
- 15 semantic categories

## 📝 Notes & Limitations

- All-pairs Jaccard similarity scales **O(n²)** with the number of cultures; for datasets larger than ~100 cultures, consider precomputing or using approximate similarity search.
- The Jaccard heatmap and radar chart become hard to read beyond ~25 cultures and ~6 cultures respectively.
- Star identifiers depend on the source catalog; cultures using different naming conventions may show artificially low similarity.
- Regional coverage is uneven — interpret cross-region comparisons with this in mind.

## 🙏 Acknowledgements

Thanks to the project client for providing the constellation datasets and guidance throughout the analysis.
Code is released under the MIT License. Data files retain the licenses specified within them.

