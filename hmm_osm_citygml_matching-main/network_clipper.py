from shapely.ops import unary_union 
import geopandas as gpd
import matplotlib.pyplot as plt
from typing import List, Tuple

class NetworkClipper:
    def __init__(
        self,
        network_gdf: gpd.GeoDataFrame,
        buffer_distance: float
    ) -> None:
        """
        Initializes the NetworkClipper with a road network GeoDataFrame and a buffer distance.

        Args:
            network_gdf (gpd.GeoDataFrame): The GeoDataFrame representing the entire road network.
            buffer_distance (float): The distance used to buffer each point sequence when clipping.
        """
        self.network_gdf = network_gdf
        self.buffer_distance = buffer_distance

    def clip_network_for_sequence(
        self,
        point_sequence: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Clips the road network for a single point sequence.

        Buffers the geometries in the point sequence by `self.buffer_distance` and returns
        the portion of the network within that buffer.

        Args:
            point_sequence (gpd.GeoDataFrame): A GeoDataFrame representing a sequence of points.

        Returns:
            gpd.GeoDataFrame: The clipped portion of the network.
        """
        sequence_geom = unary_union(point_sequence.geometry)
        sequence_buffer = sequence_geom.buffer(self.buffer_distance)
        clipped_network = self.network_gdf[self.network_gdf.intersects(sequence_buffer)]
        return clipped_network

    def clip_network_for_all_sequences(
        self,
        point_sequences: List[gpd.GeoDataFrame]
    ) -> Tuple[List[gpd.GeoDataFrame], List[gpd.GeoDataFrame]]:
        """
        Clips the network for multiple point sequences.

        Iterates over each sequence, clips the network, and appends the result to a list.
        Sequences that yield no clipped network are considered invalid and are excluded.

        Args:
            point_sequences (List[gpd.GeoDataFrame]): A list of point-sequence GeoDataFrames.

        Returns:
            Tuple[List[gpd.GeoDataFrame], List[gpd.GeoDataFrame]]: A tuple containing:
                - A list of valid point-sequence GeoDataFrames.
                - A list of corresponding clipped road-network GeoDataFrames.
        """
        clipped_networks = []
        valid_sequences = []
        for sequence in point_sequences:
            clipped_network = self.clip_network_for_sequence(sequence)
            if not clipped_network.empty:
                clipped_networks.append(clipped_network)
                valid_sequences.append(sequence)
        return valid_sequences, clipped_networks
    
    def plot_clipped_networks(
        self,
        valid_sequences: List[gpd.GeoDataFrame],
        clipped_networks: List[gpd.GeoDataFrame]
    ) -> None:
        """
        Plots the clipped networks alongside their corresponding valid point sequences.

        Args:
            valid_sequences (List[gpd.GeoDataFrame]): A list of point-sequence GeoDataFrames considered valid.
            clipped_networks (List[gpd.GeoDataFrame]): A list of clipped network GeoDataFrames.

        Returns:
            None
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        self.network_gdf.plot(ax=ax, color='black', linewidth=2)
        for clipped_network in clipped_networks:
            clipped_network.plot(ax=ax, color='red', linewidth=2)
        for sequence in valid_sequences:
            sequence.plot(ax=ax, color='green', markersize=10)
        plt.title('Clipped Networks for Point Sequences')
        plt.tight_layout()
        plt.show()