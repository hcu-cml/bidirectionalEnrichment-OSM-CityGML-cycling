import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points
from math import sqrt, exp, pi
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional

class HMMMapMatcher:
    def __init__(
        self,
        road_network_gdf: gpd.GeoDataFrame,
        max_search_distance: float = 10,
        kde_model: Optional[callable] = None
    ) -> None:
        """
        Initializes the HMMMapMatcher with a road network GeoDataFrame, a maximum search distance,
        and an optional KDE model for emission probability calculation.

        Args:
            road_network_gdf (gpd.GeoDataFrame): The GeoDataFrame representing the road network.
            max_search_distance (float, optional): Maximum distance within which to search for candidate points on the road network.
            kde_model (Optional[callable], optional): A callable KDE model that takes a distance and returns a probability density.
        """
        self.road_network_gdf = road_network_gdf
        self.max_search_distance = max_search_distance
        self.kde_model = kde_model

    def find_candidates(self, point: Point) -> List[Tuple[Point, gpd.GeoSeries]]:
        """
        Finds candidate points on the road network within the max_search_distance of the given point.

        Args:
            point (Point)

        Returns:
            List[Tuple[Point, gpd.GeoSeries]]: A list of tuples where each tuple contains the nearest point on a road segment
            and the row (as a GeoSeries) from the road network.
        """
        candidates = []
        for _, row in self.road_network_gdf.iterrows():
            line = row.geometry
            nearest_point_on_line = nearest_points(line, point)[0]
            if point.distance(nearest_point_on_line) <= self.max_search_distance:
                candidates.append((nearest_point_on_line, row))

        return candidates

    def calculate_emission_probabilities(
        self,
        point: Point,
        candidates: List[Tuple[Point, gpd.GeoSeries]]
    ) -> Dict[int, float]:
        """
        Calculates emission probabilities for each candidate based on the distance from the original point.
        If a KDE model is provided during initialization, it is used to compute probabilities; otherwise,
        a Gaussian distribution with default sigma=5.35 is used.

        Args:
            point (Point)
            candidates (List[Tuple[Point, gpd.GeoSeries]]): Candidate points (nearest points on the road) and the road segments.

        Returns:
            Dict[int, float]: A dictionary mapping candidate indices to their normalized emission probabilities.
        """
        emissions = {}
        sigma = 5.35
        for idx, candidate in enumerate(candidates):
            distance = point.distance(candidate[0])
            if self.kde_model is not None:
                probability = float(self.kde_model(distance))
            else:
                probability = (1 / (sigma * sqrt(2 * pi))) * exp(-0.5 * (distance / sigma) ** 2)
            emissions[idx] = probability
        total_prob = sum(emissions.values())
        if total_prob > 0:
            for idx in emissions:
                emissions[idx] /= total_prob
        return emissions

    def calculate_transition_probability(
        self,
        candidate_1: Point,
        candidate_2: Point,
        point_1: Point,
        point_2: Point,
        beta: float = 0.5
    ) -> float:
        """
        Calculates the transition probability between two consecutive candidate points.

        Args:
            candidate_1 (Point): The matched candidate for the previous point.
            candidate_2 (Point): The matched candidate for the current point.
            point_1 (Point): The original previous point.
            point_2 (Point): The original current point.
            beta (float, optional): Parameter controlling the effect of distance discrepancy on transition probability. Defaults to 1.0.

        Returns:
            float: The transition probability.
        """
        network_distance = candidate_1.distance(candidate_2)
        observed_distance = point_1.distance(point_2)
        distance_diff = abs(network_distance - observed_distance)
        return (1 / (2 * beta)) * exp(-distance_diff / beta)

    def calculate_initial_state_probabilities(
        self,
        candidates: List[Tuple[Point, gpd.GeoSeries]]
    ) -> Dict[int, float]:
        """
        Calculates the initial state probabilities for the given set of candidates.

        Args:
            candidates (List[Tuple[Point, gpd.GeoSeries]]): A list of candidate points for the initial observation.

        Returns:
            Dict[int, float]: A dictionary where keys are candidate indices and values are probabilities.
        """
        num_candidates = len(candidates)
        initial_probabilities = {idx: 1.0 / num_candidates for idx in range(num_candidates)}
        total_prob = sum(initial_probabilities.values())
        assert abs(total_prob - 1.0) < 1e-6, "Initial probabilities do not sum up to 1."
        return initial_probabilities

    def viterbi(self, point_sequence: gpd.GeoDataFrame) -> List[Point]:
        """
        Applies the Viterbi algorithm to find the most likely path on the network for a sequence of points.

        Args:
            point_sequence (gpd.GeoDataFrame): A GeoDataFrame containing the sequence of points.

        Returns:
            List[Point]: The most likely sequence of matched candidate points.
            Returns an empty list if no candidates exist for any point in the sequence.
        """
        point_sequence = point_sequence.geometry.to_list()
        V = [{}]
        path = {}

        # Initialize base cases (t == 0)
        first_point = point_sequence[0]
        candidates = self.find_candidates(first_point)

        # Skip the sequence if no candidates found for the first point
        if not candidates:
            print(f"No candidates found for the first point: {first_point}. Skipping this sequence.")
            return []  # Return empty path if no candidates are found for the first point

        emission_probs = self.calculate_emission_probabilities(first_point, candidates)
        initial_state_probs = self.calculate_initial_state_probabilities(candidates)

        # Initialize the first state with initial state probabilities
        for idx, candidate in enumerate(candidates):
            V[0][idx] = initial_state_probs[idx] * emission_probs[idx]
            path[idx] = [candidate[0]]

        # Run Viterbi for t > 0
        for t in range(1, len(point_sequence)):
            V.append({})
            new_path = {}

            current_point = point_sequence[t]
            previous_point = point_sequence[t - 1]
            current_candidates = self.find_candidates(current_point)
            previous_candidates = self.find_candidates(previous_point)

            # Skip the sequence if no candidates found for the current point
            if not current_candidates:
                print(f"No candidates found for point {current_point} at time {t}. Skipping this sequence.")
                return []  # Return empty path if no candidates are found for a point

            emission_probs = self.calculate_emission_probabilities(current_point, current_candidates)

            for curr_idx, curr_candidate in enumerate(current_candidates):
                max_prob = -1.0
                best_prev_idx = None

                for prev_idx, prev_candidate in enumerate(previous_candidates):
                    prob = (
                        V[t - 1][prev_idx] *
                        self.calculate_transition_probability(prev_candidate[0], curr_candidate[0], previous_point, current_point)
                    )
                    if prob > max_prob:
                        max_prob = prob
                        best_prev_idx = prev_idx

                emission_prob = emission_probs[curr_idx]
                V[t][curr_idx] = max_prob * emission_prob
                new_path[curr_idx] = path[best_prev_idx] + [curr_candidate[0]]

            path = new_path

        # For the final step, find the candidate with the highest probability
        last_time_step = len(V) - 1
        max_prob = -1.0
        best_candidate_idx = None

        for idx, prob in V[last_time_step].items():
            if prob > max_prob:
                max_prob = prob
                best_candidate_idx = idx

        return path[best_candidate_idx]
    
    def save_matched_points_with_attributes_to_gpkg(
        self,
        matched_sequences_with_attributes: List[Tuple[List[Point], gpd.GeoDataFrame]],
        output_gpkg_path: str
    ) -> None:
        """
        Saves the matched sequences (candidate points) along with original point attributes to a GeoPackage.

        Args:
            matched_sequences_with_attributes (List[Tuple[List[Point], gpd.GeoDataFrame]]): A list of tuples, where each tuple
                contains a list of matched points and the corresponding original point GeoDataFrame.
            output_gpkg_path (str): The path for the output GeoPackage.

        Returns:
            None
        """
        # List to store updated rows with matched geometries
        matched_points_with_attributes = []

        # Iterate over the matched sequences and their corresponding original point sequences
        for matched_path, point_sequence_gdf in matched_sequences_with_attributes:
            for matched_point, original_row in zip(matched_path, point_sequence_gdf.itertuples(index=False)):
                # Convert the original row attributes to a dictionary (excluding geometry)
                point_data = original_row._asdict()
                # Replace geometry with matched point
                point_data['geometry'] = Point(matched_point.x, matched_point.y)
                matched_points_with_attributes.append(point_data)

        # Create a GeoDataFrame from the list of dictionaries
        matched_gdf = gpd.GeoDataFrame(matched_points_with_attributes, crs=self.road_network_gdf.crs)

        # Save the GeoDataFrame as a GPKG file
        matched_gdf.to_file(output_gpkg_path, layer='matched_points_with_attributes', driver='GPKG')
        print(f"Saved matched points with attributes to {output_gpkg_path}")
        
    def plot_matched_sequences(
        self,
        matched_sequences_with_attributes: List[Tuple[List[Point], gpd.GeoDataFrame]]
    ) -> None:
        """
        Plots the matched sequences on top of the road network.

        Args:
            matched_sequences_with_attributes (List[Tuple[List[Point], gpd.GeoDataFrame]]): A list of tuples where each tuple
                contains a list of matched candidate points and the corresponding original point GeoDataFrame.

        Returns:
            None
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        self.road_network_gdf.plot(ax=ax, color='black', linewidth=2)
        for matched_path, point_sequence_gdf in matched_sequences_with_attributes:
            point_sequence_gdf.plot(ax=ax, color='red', markersize=10)
            matched_points = [Point(point.x, point.y) for point in matched_path]
            gpd.GeoSeries(matched_points).plot(ax=ax, color='blue', markersize=5)
        plt.title('Matched Sequences on the Road Network')
        plt.tight_layout()
        plt.show()
