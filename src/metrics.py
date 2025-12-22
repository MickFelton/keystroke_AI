import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

class KeystrokeMetrics:
    """
    Implementation of distance metrics for keystroke dynamics authentication,
    including the modified Manhattan distance metric proposed in the paper.
    """
    
    def __init__(self, du_total_weight: float = 1/3):
        """
        Initialize with the weight for DUtotal.
        
        Args:
            du_total_weight: Weight applied to DUtotal distance (paper recommends 1/3)
        """
        self.du_total_weight = du_total_weight
    
    def manhattan_distance(self, feature_vector1: pd.Series, feature_vector2: pd.Series) -> float:
        """
        Calculate the Manhattan distance between two feature vectors.
        
        This implements the standard Manhattan distance where all features have equal weight.
        
        Args:
            feature_vector1: First feature vector
            feature_vector2: Second feature vector
            
        Returns:
            Distance score (lower is more similar)
        """
        # Filter out metadata columns
        feature_cols = [col for col in feature_vector1.index 
                        if col not in ['segment_id', 'user_id']]
        
        # Calculate distances only for columns present in both vectors
        common_cols = [col for col in feature_cols 
                      if not pd.isna(feature_vector1[col]) and not pd.isna(feature_vector2[col])]
        
        if not common_cols:
            return float('inf')  # No common features
        
        # Standard Manhattan distance
        distance = 0
        for col in common_cols:
            distance += abs(feature_vector1[col] - feature_vector2[col])
        
        return distance   # No Normalization needed for Manhattan distance
    
    def modified_manhattan_distance(self, feature_vector1: pd.Series, feature_vector2: pd.Series) -> float:
        """
        Calculate the modified Manhattan distance between two feature vectors.
        
        This implements the modified Manhattan distance where DUtotal features have reduced weight.
        
        Args:
            feature_vector1: First feature vector
            feature_vector2: Second feature vector
            
        Returns:
            Distance score (lower is more similar)
        """
        # Filter out metadata columns
        feature_cols = [col for col in feature_vector1.index 
                        if col not in ['segment_id', 'user_id']]
        
        # Calculate distances only for columns present in both vectors
        common_cols = [col for col in feature_cols 
                      if not pd.isna(feature_vector1[col]) and not pd.isna(feature_vector2[col])]
        
        if not common_cols:
            return float('inf')  # No common features
        
        # Modified Manhattan distance
        distance = 0
        feature_count = 0
        
        for col in common_cols:
            diff = abs(feature_vector1[col] - feature_vector2[col])
            
            # Apply reduced weight to DUtotal features
            if '_DUtotal' in col:
                diff *= self.du_total_weight
            
            distance += diff
            feature_count += 1
        
        # print(f"Modified Distance ({feature_vector1['segment_id']} vs {feature_vector2['segment_id']}): {distance}")
        return distance # No Normalization needed for Manhattan distance