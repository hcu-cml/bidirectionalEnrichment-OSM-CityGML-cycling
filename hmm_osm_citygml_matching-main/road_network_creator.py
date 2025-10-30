import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any
import configparser

class RoadNetworkCreator:
    def __init__(
        self,
        points_layer_path: str,
        road_network_layer_path: str,
        nodes_layer_path: str
    ):
        """
        Initializes the RoadNetworkCreator with paths to the points, road network, and nodes layers.

        Args:
            points_layer_path (str): The file path to the layer containing points.
            road_network_layer_path (str): The file path to the layer containing roads.
            nodes_layer_path (str): The file path to the layer containing nodes.
        """
        self.points_layer_path = points_layer_path
        self.road_network_layer_path = road_network_layer_path
        self.nodes_layer_path = nodes_layer_path
        self.points_gdf = None
        self.network_gdf = None
        self.nodes_gdf = None
        self.node_dict: Dict[int, Any] = {}
        self.G = nx.Graph()

    def load_data(self) -> None:
        """
        Loads geospatial data from the specified file paths into GeoDataFrames.

        This method reads data for points, roads, and nodes from disk and stores
        them in the corresponding instance variables.

        Returns:
            None
        """
        self.points_gdf = gpd.read_file(self.points_layer_path)
        self.network_gdf = gpd.read_file(self.road_network_layer_path)
        self.nodes_gdf = gpd.read_file(self.nodes_layer_path)

    def ensure_crs(self) -> None:
        """
        Ensures all GeoDataFrames use the same coordinate reference system (CRS).

        Converts the points and nodes GeoDataFrames to match the CRS of the road network.

        Returns:
            None
        """
        self.points_gdf = self.points_gdf.to_crs(self.network_gdf.crs)
        self.nodes_gdf = self.nodes_gdf.to_crs(self.network_gdf.crs)

    def preprocess_data(self) -> None:
        """
        Prepares the data for building the road network graph.

        Ensures that the node identifiers (osmid) and the edge endpoints (u and v)
        are integers. Also builds a dictionary mapping node IDs to their geometries.

        Returns:
            None
        """
        self.nodes_gdf['osmid'] = self.nodes_gdf['osmid'].astype(int)
        self.network_gdf['u'] = self.network_gdf['u'].astype(int)
        self.network_gdf['v'] = self.network_gdf['v'].astype(int)
        self.node_dict = {node['osmid']: node.geometry for idx, node in self.nodes_gdf.iterrows()}

    def build_graph(self) -> None:
        """
        Constructs a NetworkX graph object from the road network GeoDataFrame.

        Iterates over each row in the road network to add nodes and edges to the
        internal NetworkX graph. Each edge is enriched with geometry and length attributes.

        Returns:
            None
        """
        for idx, row in self.network_gdf.iterrows():
            start_node_id = row['u']
            end_node_id = row['v']
            length = row['length']
            start_node_geom = self.node_dict[start_node_id]
            end_node_geom = self.node_dict[end_node_id]
            self.G.add_node(start_node_id, pos=(start_node_geom.x, start_node_geom.y))
            self.G.add_node(end_node_id, pos=(end_node_geom.x, end_node_geom.y))
            self.G.add_edge(
                start_node_id, end_node_id,
                length=length,
                start_geom=start_node_geom,
                end_geom=end_node_geom,
                geometry=row.geometry
            )

    def plot_network(self) -> None:
        """
        Plots the road network and its nodes using matplotlib.

        Displays the roads in black and the nodes in blue. This is a basic visualization
        useful for verifying the network structure.

        Returns:
            None
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        self.network_gdf.plot(ax=ax, color='black', linewidth=2)
        self.nodes_gdf.plot(ax=ax, color='blue', markersize=5)
        plt.title('Road Network')
        plt.tight_layout()
        plt.show()

    def run_creator(self) -> gpd.GeoDataFrame:
        """
        Main execution method to create and plot the road network.

        This method orchestrates data loading, CRS alignment, preprocessing,
        graph building, and an optional plot of the network.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame representing the road network.
        """
        self.load_data()
        self.ensure_crs()
        self.preprocess_data()
        self.build_graph()
        self.plot_network()
        return self.network_gdf