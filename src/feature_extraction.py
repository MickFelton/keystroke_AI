import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set
from collections import Counter

class KeystrokeFeatureExtractor:
    """
    Extracts optimal keystroke features based on the paper's findings.
    
    Key features include:
    1. Selecting the most common di-graphs (12)
    2. Using the optimal time combinations (DU1, DU2, DUtotal)
    """
    
    def __init__(self, num_digraphs: int = 12):
        """
        Initialize the feature extractor.
        
        Args:
            num_digraphs: Number of most frequent digraphs to use (default: 12 as per paper)
        """
        self.num_digraphs = num_digraphs
        self.common_digraphs = None
    
    def identify_common_digraphs(self, all_features: pd.DataFrame) -> List[str]:
        """
        Identify the most common digraphs across all segments.
        
        Args:
            all_features: DataFrame containing digraph features from all segments
            
        Returns:
            List of the most common digraphs
        """
        # Count occurrences of each digraph
        digraph_counts = Counter(all_features['digraph'])
        
        # Get the most common digraphs
        self.common_digraphs = [digraph for digraph, _ in digraph_counts.most_common(self.num_digraphs)]
        
        return self.common_digraphs
    
    def create_feature_vectors(self, all_features: pd.DataFrame) -> pd.DataFrame:
        """
        Create feature vectors for each segment, using the most common digraphs.
        
        For each segment and each common digraph, calculate the average DU1, DU2, and DUtotal.
        
        Args:
            all_features: DataFrame containing digraph features from all segments
            
        Returns:
            DataFrame with feature vectors for each segment
        """
        if self.common_digraphs is None:
            self.identify_common_digraphs(all_features)
        
        # Initialize results dictionary
        results = []
        
        # Process each segment
        for segment_id, segment_data in all_features.groupby('segment_id'):
            user_id = segment_data['user_id'].iloc[0]
            
            # Create feature vector for this segment
            feature_vector = {'segment_id': segment_id, 'user_id': user_id}
            
            # Add features for each common digraph
            for digraph in self.common_digraphs:
                # Get rows for this digraph in this segment
                digraph_rows = segment_data[segment_data['digraph'] == digraph]
                
                if len(digraph_rows) > 0:
                    # Calculate average timing values
                    feature_vector[f"{digraph}_DU1"] = digraph_rows['DU1'].median()
                    feature_vector[f"{digraph}_DU2"] = digraph_rows['DU2'].median()
                    feature_vector[f"{digraph}_DUtotal"] = digraph_rows['DUtotal'].median()
                else:
                    # If digraph not present in this segment, use NaN
                    feature_vector[f"{digraph}_DU1"] = np.nan
                    feature_vector[f"{digraph}_DU2"] = np.nan
                    feature_vector[f"{digraph}_DUtotal"] = np.nan
            
            results.append(feature_vector)
        
        df = pd.DataFrame(results)
        # # Debug: print NaN stats
        # print("NaN stats per feature vector (first 10):")
        # print(df.isna().sum(axis=1).head(10))
        # print("Total vectors:", len(df))
        # print("Vectors with all NaN features:", (df.drop(['segment_id', 'user_id'], axis=1).isna().all(axis=1)).sum())
        # # Convert to DataFrame
        return df