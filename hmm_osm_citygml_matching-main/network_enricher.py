import logging
import matplotlib.pyplot as plt
from shapely.geometry import Point

class NetworkEnricher:
    def __init__(self, network_gdf):
        """
        Initialize the NetworkEnricher with the original road network.
        :param network_gdf: GeoDataFrame of the original road network.
        """
        self.network_gdf = network_gdf.copy()

    def enrich_network_with_matched_attributes(self, matched_sequences_with_attributes, buffer_distance=0.001):
        """
        Enrich the road network by transferring attributes from matched sequences to intersecting road segments.
        :param matched_sequences_with_attributes: List of tuples containing matched paths and their original point sequences.
        :param buffer_distance: Buffer distance to create around each matched point.
        :return: Enriched road network GeoDataFrame.
        """
        # Iterate over matched paths and the corresponding original rows in point_sequences_gdf
        for matched_path, point_sequence_gdf in matched_sequences_with_attributes:
            for matched_point in matched_path:
                matched_point_geom = Point(matched_point.x, matched_point.y)  # Convert to shapely Point

                # Apply a small buffer to the matched point to handle precision issues
                buffered_point_geom = matched_point_geom.buffer(buffer_distance)

                # Find the road segment(s) that intersect with the buffered point
                intersecting_segments = self.network_gdf[self.network_gdf.intersects(buffered_point_geom)]

                # Transfer attributes to each intersecting segment
                if not intersecting_segments.empty:
                    # Assuming all points in the matched path come from the same original point sequence
                    original_row = point_sequence_gdf.iloc[0] 
                    point_data = original_row.to_dict()
                    if 'geometry' in point_data:
                        del point_data['geometry']  # Do not override the geometry of the road segment

                    for idx, road_segment in intersecting_segments.iterrows():
                        # Add these attributes to the road segment
                        for key, value in point_data.items():
                            if key not in self.network_gdf.columns:
                                # Add the new attribute if it doesn't exist
                                self.network_gdf.loc[idx, key] = value
                            else:
                                # Handle attribute conflict (e.g., you can average or overwrite)
                                self.network_gdf.loc[idx, key] = value  # Overwrite for now

        logging.info("Enriched network_gdf with matched points' attributes.")
        return self.network_gdf

    def save_enriched_network(self, enriched_network_gdf, output_path):
        """
        Save the enriched network to a GeoPackage file.
        :param enriched_network_gdf: Enriched road network GeoDataFrame.
        :param output_path: File path to save the GeoPackage.
        """
        enriched_network_gdf.to_file(output_path, layer='enriched_network', driver='GPKG')
        logging.info(f"Enriched network saved to {output_path}")

    def plot_enriched_network(self, enriched_network_gdf):
        """
        Optional: Plot the enriched network for visualization.
        :param enriched_network_gdf: Enriched road network GeoDataFrame.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        enriched_network_gdf.plot(ax=ax, column='your_attribute_column', legend=True)
        plt.title('Enriched Road Network')
        plt.tight_layout()
        plt.show()