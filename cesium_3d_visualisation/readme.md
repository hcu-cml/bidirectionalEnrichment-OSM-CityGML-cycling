# Enriched CityGML Visualization with Cesium

This project visualizes enriched CityGML data, combining 3D city models with bicycle infrastructure scores in an interactive CesiumJS viewer. The goal is to enable intuitive inspection of spatial safety information through web-based 3D visualization.

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
