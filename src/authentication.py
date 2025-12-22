import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from .metrics import KeystrokeMetrics

class KeystrokeAuthenticator:
    """
    Implements keystroke dynamics authentication using the leave-one-out methodology.
    """
    
    def __init__(self, du_total_weight: float = 1/3):
        """
        Initialize the authenticator.
        
        Args:
            du_total_weight: Weight for DUtotal in modified Manhattan distance (default: 1/3)
        """
        self.metrics = KeystrokeMetrics(du_total_weight=du_total_weight)
    
    def leave_one_out_evaluation(self, feature_vectors: pd.DataFrame, 
                               use_modified_distance: bool = True) -> Dict:
        """
        Perform leave-one-out evaluation on the feature vectors.
        
        For each feature vector, compare against all others and calculate FAR/FRR.
        
        Args:
            feature_vectors: DataFrame with feature vectors for each segment
            use_modified_distance: Whether to use modified Manhattan distance
            
        Returns:
            Dictionary with distance matrices and user information
        """
        n_segments = len(feature_vectors)
        distance_matrix = np.zeros((n_segments, n_segments))
        
        # Get user IDs for each segment
        segment_ids = feature_vectors['segment_id'].values
        user_ids = feature_vectors['user_id'].values
        
        # Calculate distance matrix
        for i in range(n_segments):
            for j in range(i+1, n_segments):  # Only need upper triangle
                if use_modified_distance:
                    dist = self.metrics.modified_manhattan_distance(
                        feature_vectors.iloc[i], feature_vectors.iloc[j])
                else:
                    dist = self.metrics.manhattan_distance(
                        feature_vectors.iloc[i], feature_vectors.iloc[j])
                
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist  # Matrix is symmetric
               # Debug: print some distances
        # print("Sample distances (first 5x5):")
        # print(distance_matrix[:5, :5])
        # print("Any inf in distance matrix?", np.isinf(distance_matrix).any())
        # print("Any NaN in distance matrix?", np.isnan(distance_matrix).any())
        
        return {
            'distance_matrix': distance_matrix,
            'segment_ids': segment_ids,
            'user_ids': user_ids
        }
    
    def calculate_error_rates(self, evaluation_results: Dict, thresholds: np.ndarray) -> Dict:
        """
        Calculate FAR, FRR, and EER for different thresholds.
        
        Args:
            evaluation_results: Results from leave_one_out_evaluation
            thresholds: Array of threshold values to evaluate
            
        Returns:
            Dictionary with FAR, FRR for each threshold, and EER
        """
        distance_matrix = evaluation_results['distance_matrix']
        user_ids = evaluation_results['user_ids']
        
        n_segments = len(user_ids)
        results = {
            'thresholds': thresholds,
            'far': [],
            'frr': []
        }
        
        # For each threshold
        for threshold in thresholds:
            false_accepts = 0
            false_rejects = 0
            total_genuine = 0
            total_impostor = 0
            
            # Compare each segment with all others
            for i in range(n_segments):
                for j in range(n_segments):
                    if i == j:
                        continue  # Skip comparing segment with itself
                    
                    is_same_user = user_ids[i] == user_ids[j]
                    distance = distance_matrix[i, j]
                    authenticated = distance <= threshold
                    
                    if is_same_user:
                        # Genuine attempt
                        total_genuine += 1
                        if not authenticated:
                            false_rejects += 1
                    else:
                        # Impostor attempt
                        total_impostor += 1
                        if authenticated:
                            false_accepts += 1
            
            # Calculate rates
            far = false_accepts / total_impostor if total_impostor > 0 else 0
            frr = false_rejects / total_genuine if total_genuine > 0 else 0
            
            results['far'].append(far)
            results['frr'].append(frr)
        
        # Find EER (where FAR=FRR)
        differences = np.abs(np.array(results['far']) - np.array(results['frr']))
        eer_index = np.argmin(differences)
        eer = (results['far'][eer_index] + results['frr'][eer_index]) / 2
        eer_threshold = thresholds[eer_index]
        
        results['eer'] = eer
        results['eer_threshold'] = eer_threshold
        
        return results