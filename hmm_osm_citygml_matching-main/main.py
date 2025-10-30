import configparser
import logging
import os
import geopandas as gpd

from road_network_creator import RoadNetworkCreator
from point_sequence_generator import PointSequenceGenerator
from network_clipper import NetworkClipper
from hmm_map_matcher import HMMMapMatcher 
from network_enricher import NetworkEnricher

from calculating_safety_scores import process_and_save_scores
from kde_utils import load_kde, plot_kde
import numpy as np

def setup_logging():
    """
    Sets up the logging configuration for the application.

    This function configures the logging level to INFO and sets a specific format for log messages.
    It doesn't take any arguments and doesn't return any value.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(config_file='config.ini'):
    """
    Loads the configuration from an .ini file using ConfigParser.

    Args:
        config_file (str): The path to the .ini configuration file. Defaults to 'config.ini'.

    Returns:
        configparser.ConfigParser: The loaded configuration object.
    """
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


def ensure_directories(output_paths):
    """
    Ensures that the directories for the provided output paths exist.

    Args:
        output_paths (list[str]): A list of file paths for which directories need to be created.

    Returns:
        None
    """
    for path in output_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def run_network_creation(config: configparser.ConfigParser) -> gpd.GeoDataFrame:
    """
    Instantiates and runs the RoadNetworkCreator to produce a road network GeoDataFrame.

    Args:
        config (configparser.ConfigParser): The configuration object containing the paths.

    Returns:
        gpd.GeoDataFrame: The generated road network.
    """
    points_layer_path = config['paths']['points_layer_path']
    road_network_layer_path = config['paths']['road_network_layer_path']
    nodes_layer_path = config['paths']['nodes_layer_path']

    road_network_creator = RoadNetworkCreator(points_layer_path, road_network_layer_path, nodes_layer_path)
    network_gdf = road_network_creator.run_creator()
    return network_gdf


def run_point_sequence_generation(
    config: configparser.ConfigParser,
    network_gdf: gpd.GeoDataFrame
) -> tuple[list[gpd.GeoDataFrame], PointSequenceGenerator]:
    """
    Instantiates and runs the PointSequenceGenerator to produce sequences of points.

    Args:
        config (configparser.ConfigParser): The configuration object containing the generator parameters.
        network_gdf (gpd.GeoDataFrame): The GeoDataFrame containing the road network.

    Returns:
        tuple[list[gpd.GeoDataFrame], PointSequenceGenerator]: A tuple with the generated point sequences and the generator instance.
    """
    points_layer_path = config['paths']['points_layer_path']
    expected_distance = float(config['point_sequence_generator']['expected_distance'])
    max_gap_distance = float(config['point_sequence_generator']['max_gap_distance'])

    point_generator = PointSequenceGenerator(points_layer_path, expected_distance, max_gap_distance)
    point_generator.load_points()
    point_generator.ensure_crs(network_gdf.crs)
    point_generator.group_points_by_distance()
    point_sequences = point_generator.get_sequences()
    
    # Uncomment below lines if you want to save sequences to GPKG
    # output_gpkg_path_sequences = config['paths']['output_gpkg_path_sequences']
    # point_generator.save_sequences_to_gpkg(output_gpkg_path_sequences)
    
    point_generator.plot_sequences(network_gdf)
    
    return point_sequences, point_generator 


def run_network_clipping(
    config: configparser.ConfigParser,
    network_gdf: gpd.GeoDataFrame,
    point_sequences: list[gpd.GeoDataFrame]
) -> tuple[list[gpd.GeoDataFrame], list[gpd.GeoDataFrame]]:
    """
    Instantiates and runs the NetworkClipper for each sequence.

    Args:
        config (configparser.ConfigParser): The configuration object containing the clipping parameters.
        network_gdf (gpd.GeoDataFrame): The original road network.
        point_sequences (list[gpd.GeoDataFrame]): A list of point-sequence GeoDataFrames.

    Returns:
        tuple[list[gpd.GeoDataFrame], list[gpd.GeoDataFrame]]: A tuple with the valid sequences and their clipped networks.
    """
    buffer_distance = float(config['network_clipper']['buffer_distance'])
    
    clipper = NetworkClipper(network_gdf, buffer_distance)
    valid_sequences, clipped_networks = clipper.clip_network_for_all_sequences(point_sequences)
    clipper.plot_clipped_networks(valid_sequences, clipped_networks)
    
    return valid_sequences, clipped_networks


def run_map_matching(
    config: configparser.ConfigParser,
    valid_sequences: list[gpd.GeoDataFrame],
    clipped_networks: list[gpd.GeoDataFrame],
    kde_model=None
) -> tuple[list[tuple[list[int], gpd.GeoDataFrame]], HMMMapMatcher]:
    """
    Runs the HMM-based map matching for each point sequence and clipped network.

    Args:
        config (configparser.ConfigParser): The configuration object containing the output paths.
        valid_sequences (list[gpd.GeoDataFrame]): A list of valid point-sequence GeoDataFrames.
        clipped_networks (list[gpd.GeoDataFrame]): A list of clipped road-network GeoDataFrames.

    Returns:
        tuple[list[tuple[list[int], gpd.GeoDataFrame]], HMMMapMatcher]: A tuple containing matched sequences with attributes and the HMMMapMatcher instance.
    """
    matched_sequences_with_attributes = []
    
    for point_sequence_gdf, clipped_network in zip(valid_sequences, clipped_networks):
        hmm_matcher = HMMMapMatcher(clipped_network, kde_model=kde_model)
        matched_path = hmm_matcher.viterbi(point_sequence_gdf)
        if matched_path:
            matched_sequences_with_attributes.append((matched_path, point_sequence_gdf))
        logging.info(f"Matched Path: {matched_path}")
    
    # Save matched points with attributes
    output_gpkg_path_matched_points = config['paths']['output_gpkg_path_matched_points']
    hmm_matcher.save_matched_points_with_attributes_to_gpkg(matched_sequences_with_attributes, output_gpkg_path_matched_points)
    hmm_matcher.plot_matched_sequences(matched_sequences_with_attributes)
    
    return matched_sequences_with_attributes, hmm_matcher


def run_network_enrichment(
    config: configparser.ConfigParser,
    matched_sequences_with_attributes: list[tuple[list[int], gpd.GeoDataFrame]],
    original_network_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Enriches the original road network with attributes from matched sequences, saving the result as a GPKG file.

    Args:
        config (configparser.ConfigParser): The configuration object containing the output path.
        matched_sequences_with_attributes (list[tuple[list[int], gpd.GeoDataFrame]]): A list of matched paths and the original point sequence.
        original_network_gdf (gpd.GeoDataFrame): The original, unmodified road network GeoDataFrame.

    Returns:
        gpd.GeoDataFrame: The enriched road network.
    """
    output_gpkg_path_enriched_network = config['paths']['output_gpkg_path_enriched_network']
    
    # Initialize the NetworkEnricher
    enricher = NetworkEnricher(original_network_gdf)
    
    # Enrich the network
    enriched_network_gdf = enricher.enrich_network_with_matched_attributes(matched_sequences_with_attributes, buffer_distance=0.001)
    
    # Save the enriched network
    enricher.save_enriched_network(enriched_network_gdf, output_gpkg_path_enriched_network)
    
    # Optional: Plot the enriched network
    # enricher.plot_enriched_network(enriched_network_gdf)
    
    return enriched_network_gdf


def main():
    """
    Main function to orchestrate the entire workflow.

    This function:
    - Loads the configuration
    - Ensures output directories exist
    - Runs the network creation
    - Generates point sequences
    - Clips the network
    - Performs HMM map matching
    - Enriches the network

    Logs progress and handles any exceptions that occur.
    """
    setup_logging()

    # Load config
    config = load_config()

    # --- KDE emission model (optional) ---
    kde_model = load_kde(config['paths'].get('residual_distance_npy', 'distances.npy'))
    if kde_model is None:
        logging.warning("No residual distance file found; HMM will fall back to Gaussian emission.")
    else:
        logging.info("KDE model loaded successfully.")

        # Optional: visualise KDE once at startup
        try:
            dist_path = config['paths'].get('residual_distance_npy', 'distances.npy')
            d_vis = np.load(dist_path)
            plot_kde(kde_model, d_vis, max_x=10, save_path="output/kde_hist_gaussian.pdf")
        except Exception as ex:
            logging.warning(f"KDE visualisation skipped: {ex}")

    # Ensure output directories exist
    ensure_directories([
        config['paths'][k] for k in (
            'output_gpkg_path_sequences',
            'output_gpkg_path_clipped',
            'output_gpkg_path_matched_points',
            'output_gpkg_path_enriched_network'
        )
    ])

    # Run workflow
    try:
        # Step 1: Network Creation
        network_gdf = run_network_creation(config)

        # Step 2: Point Sequence Generation
        point_sequences, point_generator = run_point_sequence_generation(config, network_gdf)

        # Step 3: Network Clipping
        valid_sequences, clipped_networks = run_network_clipping(config, network_gdf, point_sequences)

        # Step 4: Map Matching
        matched_sequences_with_attributes, hmm_matcher = run_map_matching(
            config, valid_sequences, clipped_networks, kde_model
        )

        # Step 5: Network Enrichment
        enriched_network_gdf = run_network_enrichment(config, matched_sequences_with_attributes, network_gdf)

        # Step 6: Calculate Safety Scores
        process_and_save_scores()

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)


if __name__ == '__main__':
    main()