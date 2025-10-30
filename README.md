This repository implements the core modules for the bidirectional enrichment of OpenStreetMap (OSM) and CityGML data for cycling safety assessment. The goal is to map match and accomplish an attribute transfer between OSM and CityGML as described in the accompanying paper. After the attribute transfer, a regression model is applied using two sets of training data to calculate dual cycling safety scores: one that incorporates width information and one that omits it. These scores are appended to the enriched network. The workflow includes:
1. **Road Network Creator:**  
   *Input:* OSM nodes and edges layers (`road_network_layer_path`, `nodes_layer_path`)  
   *Output:* `network_gdf` — A GeoDataFrame representing the full OSM-based road network, including nodes and edges, suitable for spatial and topological processing.

2. **Point Sequence Generator:**  
   *Input:* Preprocessed CityGML point layer (`points_layer_path`)  
   *Output:* `point_sequences` — A list of GeoDataFrames, each containing an ordered sequence of points representing a potential segment derived from CityGML data.

3. **Network Clipper:**  
   *Input:* Full road network (`network_gdf`) and point sequences (`point_sequences`)  
   *Output:*  
   `valid_sequences` — A list of point sequences that had at least one intersecting road segment within the specified buffer distance.  
   `clipped_networks` — A list of corresponding clipped road networks (GeoDataFrames), each containing only the road segments spatially near a valid point sequence.

4. **HMM Map Matcher (KDE-based):**  
   *Input:* Clipped networks (`clipped_networks`) and point sequences (`valid_sequences`)  
   *Output:* `matched_sequences_with_attributes` — A list of tuples containing matched path indices and the corresponding original point sequence.  
   *Details:* Emission probabilities are computed using a Kernel Density Estimation (KDE) model fit to the orthogonal distances between CityGML centreline samples and nearby OSM candidates (within a 10 m search window). This non‑parametric approach captures skewed, heavy‑tailed residuals at complex junctions better than a Gaussian assumption, improving robustness. Bandwidth is selected via Silverman’s rule; transitions follow a standard HMM/Viterbi formulation comparing network path length to Euclidean step length.

5. **Network Enricher:**  
   *Input:* Full road network (`network_gdf`) and matched sequences (`matched_sequences_with_attributes`)  
   *Output:* `enriched_network_gdf` — A GeoDataFrame where the original road network has been enriched with transferred attributes (e.g. from CityGML).

6. **Cycling Safety Score Calculation:**  
   *Input:* Enriched road network (`enriched_network.gpkg`)  
   *Output:* `enriched_network_with_dual_scores.gpkg` — A GeoPackage file that includes safety scores for each road segment, computed using models that consider and ignore width information.

```mermaid
flowchart TD
    A[OSM Nodes and Edges] --> B[Road Network Creator: network_gdf]
    F[CityGML Points] --> C[Point Sequence Generator: point_sequences]
    
    B --> D[Network Clipper: valid_sequences, clipped_networks]
    C --> D

    D --> E[HMM Map Matcher (KDE): matched_sequences_with_attributes]
    E --> G[Network Enricher: enriched_network.gpkg]
    G --> H[Safety Score Calculator: enriched_network_with_dual_scores.gpkg]
```

## Installation

### Requirements

- **Python Version:** 3.7 or higher  
- **Dependencies:**  
    - configparser==7.1.0
    - geopandas==1.0.1
    - matplotlib==3.9.2
    - networkx==3.3
    - numpy==2.1.1
    - scipy==1.14.1
    - shapely==2.0.6
    - scikit-learn==1.6.1

It is recommended to use a virtual environment.

### Setup Instructions

The github link is currently unavailable due to the double-blind rule.
You can get the entire project from the figshare link: https://figshare.com/s/323766fd1f84c797cf02


1. **Clone the Repository:**
   ```bash
   git clone https://github.com/username/hmm_osm_citygml_matching.git
   cd hmm_osm_citygml_matching

2.    Create and Activate a Virtual Environment (optional but recommended):

- On Linux/Mac
    ```bash
    python3 -m venv venv
    source venv/bin/activate

 - On Windows: 
    ```bash
     python -m venv venv
    venv\Scripts\activate


3.    Install Dependencies:
    ```bash
    pip install -r requirements.txt

Run the main script:

python main.py

Final output:
- `enriched_network_with_dual_scores.gpkg`: GeoPackage containing the final scored cycling network.

## citygml-3d-visualization-main

This part visualizes enriched CityGML data, combining 3D city models with bicycle infrastructure scores in an interactive CesiumJS viewer. The goal is to enable intuitive inspection of spatial safety information through web-based 3D visualization.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Run the Server](#run-the-server)
- [Usage](#usage)

## Prerequisites

- Node.js (version 14 or later): https://nodejs.org
- A modern web browser

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/username/enriched_citygml_cesium.git
cd enriched_citygml_cesium
npm install
```

## Run the Server

Start the Server locally:
```bash 
node server.js
```
You should see:
```bash
Application Running: http://localhost:3000
```

## Usage:

Open a browser and navigate to:
```bash
http://localhost:3000
```

You will see:
- A basemap (OpenStreetMap via Carto)
- City buildings (tiles_city)
- Bicycle infrastructure with safety score-based color styling (cycle_area_new_score)
- External Hamburg LoD3 tilesets loaded directly via URL
- Custom terrain from the Hamburg GDI3D server

Click on any tile feature of the bicycle infrastructure to view its properties in an information box.

| Property Key              | Description                                  |
|---------------------------|----------------------------------------------|
| `score_with_width`        | Computed safety score                        |
| `_width`                  | Width of the bike lane                       |
| `_slope_percent`          | Gradient in percentage                       |
| `maxspeed`                | Road speed limit (in km/h)                   |
| `lanes`                   | Number of lanes                              |
| `parking`                 | Whether parking is present (true/false)      |
| `opendrive_road_junction` | Whether part of a road junction (true/false) |
