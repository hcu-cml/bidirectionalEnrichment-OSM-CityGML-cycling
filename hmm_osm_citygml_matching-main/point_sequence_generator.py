import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

class PointSequenceGenerator:
    def __init__(
        self,
        points_layer_path: str,
        expected_distance: float = 1.0,
        max_gap_distance: float = 1.0001
    ) -> None:
        """
        Initializes the PointSequenceGenerator with a points layer path and distance thresholds.

        Args:
            points_layer_path (str): Path to the file containing point geometries.
            expected_distance (float, optional): The expected distance between consecutive points. Defaults to 1.0.
            max_gap_distance (float, optional): Maximum allowed gap distance before a new sequence is started. Defaults to 1.5.
        """
        self.points_layer_path = points_layer_path
        self.expected_distance = expected_distance
        self.max_gap_distance = max_gap_distance
        self.points_gdf: Optional[gpd.GeoDataFrame] = None
        self.point_sequences: List[gpd.GeoDataFrame] = []

    def load_points(self) -> None:
        """
        Loads point data from the specified file path into a GeoDataFrame.

        Returns:
            None
        """
        self.points_gdf = gpd.read_file(self.points_layer_path)

    def ensure_crs(self, target_crs: str) -> None:
        """
        Ensures that the points GeoDataFrame has the specified CRS.

        Args:
            target_crs (str): The target coordinate reference system (CRS) to convert to.

        Returns:
            None
        """
        self.points_gdf = self.points_gdf.to_crs(target_crs)

    def group_points_by_distance(self) -> None:
        """
        Groups points into sequences based on the distance between consecutive points.

        If the distance between two consecutive points exceeds `max_gap_distance`,
        a new sequence is started. The results are stored in `self.point_sequences`.

        Returns:
            None
        """
        self.points_gdf = self.points_gdf[~self.points_gdf.is_empty]
        sequences = []
        current_sequence_indices = [self.points_gdf.index[0]]
        for i in range(1, len(self.points_gdf)):
            current_index = self.points_gdf.index[i]
            previous_index = self.points_gdf.index[i - 1]
            current_point = self.points_gdf.loc[current_index]
            previous_point = self.points_gdf.loc[previous_index]
            dist = current_point.geometry.distance(previous_point.geometry)
            if dist <= self.max_gap_distance:
                current_sequence_indices.append(current_index)
            else:
                sequences.append(self.points_gdf.loc[current_sequence_indices])
                current_sequence_indices = [current_index]
        if current_sequence_indices:
            sequences.append(self.points_gdf.loc[current_sequence_indices])
        self.point_sequences = sequences

    def get_sequences(self) -> List[gpd.GeoDataFrame]:
        """
        Retrieves the list of point sequences.

        Returns:
            List[gpd.GeoDataFrame]: A list of GeoDataFrames, each representing a sequence of points.
        """
        return self.point_sequences
    
    def plot_sequences(self, network_gdf: gpd.GeoDataFrame) -> None:
        """
        Plots the generated point sequences on top of the provided network GeoDataFrame.

        Args:
            network_gdf (gpd.GeoDataFrame): A GeoDataFrame representing the road network.

        Returns:
            None
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        network_gdf.plot(ax=ax, color='black', linewidth=2)
        for sequence in self.point_sequences:
            sequence.plot(ax=ax, color='red', markersize=10)
        plt.title('Point Sequences on the Network')
        plt.tight_layout()
        plt.show()

    def save_sequences_to_gpkg(self, output_gpkg_path: str) -> None:
        """
        Saves each point sequence to a separate layer within a GeoPackage file.

        Args:
            output_gpkg_path (str): File path for the output GeoPackage.

        Returns:
            None
        """
        for i, sequence in enumerate(self.point_sequences):
            sequence_gdf = gpd.GeoDataFrame(geometry=sequence, crs=self.points_gdf.crs)
            layer_name = f"sequence_{i}"
            sequence_gdf.to_file(output_gpkg_path, layer=layer_name, driver="GPKG")
        print(f"Saved {len(self.point_sequences)} sequences to {output_gpkg_path}.")