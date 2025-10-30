"""
This script processes an enriched road network to compute safe cycling scores.
It applies transformations and predictions using trained adjustment models to generate dual safety scores,
taking into account the presence or absence of width information. Results are saved to a GPKG file.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import os
from adjustment import Adjustment

def convert_to_numeric(gdf):
    """
    Converts relevant columns in the GeoDataFrame to numeric format and applies width filtering.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame with road network data.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with '_width' and '_slope_percent' converted and cleaned.
    """
    gdf['_width'] = pd.to_numeric(gdf['_width'], errors='coerce')
    gdf['_slope_percent'] = pd.to_numeric(gdf['_slope_percent'], errors='coerce')
    gdf.loc[gdf['_width'] < 0.5, '_width'] = np.nan
    gdf.loc[gdf['_width'] > 4, '_width'] = np.nan
    return gdf


def min_max_nomination(gdf):
    """
    Normalizes the '_width' column to a 0-1 scale and stores the result in 'width_score'.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing the '_width' column.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with the added 'width_score' column.
    """
    gdf['width_score'] = (gdf['_width'] - 0) / (4 - 0)
    return gdf


def calculate_dual_scores(gdf):
    """
    Calculates safety scores for different types of bike lanes using models with and without width as a feature.

    This function:
    - Loads two models (one with width, one without),
    - Applies predictions for each relevant bike lane type,
    - Computes the delta between the two score variants,
    - Applies fixed rules for special lane categories.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame enriched with necessary input features.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with 'score_with_width', 'score_without_width', and 'delta_score' columns added.
    """
    # Load models
    model_with_width = Adjustment("training_data_width.csv")
    model_without_width = Adjustment("training_data.csv")

    model_with_width.fit()
    model_without_width.fit()

    full_features = [col for col in ['maxspeed_score', 'lane_score_normalized', 'parking_score', 'crossing_score', 'width_score'] if col in gdf.columns]
    reduced_features = [col for col in ['maxspeed_score', 'lane_score_normalized', 'parking_score', 'crossing_score'] if col in gdf.columns]

    invalid_lanes = (gdf['independent_lane'] == 1) | (gdf['banned_lane'] == 1) | (gdf['push_lane'] == 1)
    bike_types = [col for col in ['cycle_lane', 'safety_lane', 'separated_lane'] if col in gdf.columns and gdf[col].sum() > 0]

    for bike_type in bike_types:
        type_mask = gdf[bike_type] == 1
        valid_rows = (~invalid_lanes) & type_mask

        # With width
        df_with = gdf.loc[valid_rows, full_features].dropna()
        if not df_with.empty:
            idx_with = df_with.index
            preds_with = model_with_width.predict(bike_type, df_with.values)
            gdf.loc[idx_with, 'score_with_width'] = preds_with

        # Without width
        df_without = gdf.loc[valid_rows, reduced_features].dropna()
        if not df_without.empty:
            idx_without = df_without.index
            preds_without = model_without_width.predict(bike_type, df_without.values)
            gdf.loc[idx_without, 'score_without_width'] = preds_without

    gdf['delta_score'] = gdf['score_with_width'] - gdf['score_without_width']

    # Special lane rules
    gdf.loc[gdf['independent_lane'] == 1, ['score_with_width', 'score_without_width', 'delta_score']] = 0
    gdf.loc[gdf['push_lane'] == 1, ['score_with_width', 'score_without_width', 'delta_score']] = 0
    gdf.loc[gdf['banned_lane'] == 1, ['score_with_width', 'score_without_width']] = 1
    gdf.loc[gdf['banned_lane'] == 1, 'delta_score'] = 0

    return gdf

def process_and_save_scores(input_path="output/enriched_network.gpkg", output_path="output/enriched_network_with_dual_scores.gpkg"):
    """
    Main function to run the full safety scoring pipeline from loading to saving.

    This function:
    - Loads the enriched network,
    - Cleans and transforms data,
    - Calculates dual scores,
    - Saves the resulting GeoDataFrame to a GPKG file.

    Args:
        input_path (str): Path to the input enriched network GPKG file.
        output_path (str): Path to save the output GPKG with dual scores.

    Returns:
        str: Path to the saved output file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    gdf = gpd.read_file(input_path)
    logging.info("Loading enriched network...")
    gdf = convert_to_numeric(gdf)
    logging.info("Converting to numeric...")
    gdf = min_max_nomination(gdf)
    logging.info("Normalizing width...")
    gdf = calculate_dual_scores(gdf)
    logging.info("Calculating dual scores...")
    gdf.to_file(output_path, driver="GPKG")
    logging.info(f"Saving to {output_path}")
    return output_path