import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

from src.data_processing import KeystrokeProcessor
from src.feature_extraction import KeystrokeFeatureExtractor
from src.authentication import KeystrokeAuthenticator

def process_data():
    """Process raw data and create feature vectors."""
    # Process raw keystroke data
    processor = KeystrokeProcessor()
    
    # Check if processed data already exists
    if os.path.exists(os.path.join(processor.processed_data_dir, 'all_features.csv')):
        print("Loading processed keystroke features...")
        all_features = pd.read_csv(os.path.join(processor.processed_data_dir, 'all_features.csv'))
    else:
        print("Processing raw keystroke data...")
        all_features = processor.process_all_users()
    
    # Print statistics
    n_segments = len(all_features['segment_id'].unique())
    n_users = len(all_features['user_id'].unique())
    print(f"Processed {n_segments} segments from {n_users} users")
    print(f"Total of {len(all_features)} digraph entries")

    # Extract feature vectors
    extractor = KeystrokeFeatureExtractor(num_digraphs=12)
    extractor.identify_common_digraphs(all_features)
    feature_vectors = extractor.create_feature_vectors(all_features)
    
    # Create two differently normalized versions
    decimal_vectors = decimal_scaling(feature_vectors.copy())
    minmax_vectors = min_max_scaling(feature_vectors.copy())
    
    # Save the feature vectors
    decimal_vectors.to_csv(os.path.join(processor.processed_data_dir, 'decimal_vectors.csv'), index=False)
    minmax_vectors.to_csv(os.path.join(processor.processed_data_dir, 'minmax_vectors.csv'), index=False)
    print(f"Saved normalized feature vectors")
    
    return decimal_vectors, minmax_vectors

def decimal_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Apply decimal scaling normalization."""
    normalized = df.copy()
    feature_cols = [col for col in df.columns if col not in ['segment_id', 'user_id']]
    
    for col in feature_cols:
        # Find maximum absolute value
        max_abs = df[col].abs().max()
        
        # Determine number of digits
        if max_abs > 0:
            digits = int(np.floor(np.log10(max_abs))) + 1
            normalized[col] = df[col] / (10 ** digits)
    
    print("Features normalized using Decimal scaling")
    return normalized

def min_max_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Apply min-max normalization to feature columns."""
    normalized = df.copy()
    feature_cols = [col for col in df.columns if col not in ['segment_id', 'user_id']]
    
    for col in feature_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        
        # Avoid division by zero
        if max_val > min_val:
            normalized[col] = (df[col] - min_val) / (max_val - min_val)
    
    print("Features normalized using Min-Max scaling")
    return normalized

def compare_distance_metrics(decimal_vectors: pd.DataFrame, minmax_vectors: pd.DataFrame):
    """Compare standard and modified Manhattan distance metrics."""
    # Create authenticators with different distance metrics
    standard_authenticator = KeystrokeAuthenticator(du_total_weight=1.0)
    modified_authenticator = KeystrokeAuthenticator(du_total_weight=1/3)
    
    # Perform leave-one-out evaluation with different normalizations
    print("Evaluating standard Manhattan distance with decimal scaling...")
    standard_eval = standard_authenticator.leave_one_out_evaluation(
        decimal_vectors, use_modified_distance=False)
    
    print("Evaluating modified Manhattan distance with min-max scaling...")
    modified_eval = modified_authenticator.leave_one_out_evaluation(
        minmax_vectors, use_modified_distance=True)
    
    # Calculate error rates for different thresholds - using separate threshold ranges
    # For standard (decimal scaled)
    std_matrix = standard_eval['distance_matrix']
    std_min = np.min(std_matrix[std_matrix > 0])
    std_max = np.max(std_matrix)
    standard_thresholds = np.linspace(std_min, std_max, 1000)
    print(f"Standard distance matrix range: {std_min:.4f} to {std_max:.4f}")
    
    # For modified (min-max scaled)
    mod_matrix = modified_eval['distance_matrix']
    mod_min = np.min(mod_matrix[mod_matrix > 0])
    mod_max = np.max(mod_matrix)
    modified_thresholds = np.linspace(mod_min, mod_max, 1000)
    print(f"Modified distance matrix range: {mod_min:.4f} to {mod_max:.4f}")
    
    # Calculate error rates with appropriate thresholds
    standard_results = standard_authenticator.calculate_error_rates(standard_eval, standard_thresholds)
    modified_results = modified_authenticator.calculate_error_rates(modified_eval, modified_thresholds)

    print("\nStandard Manhattan Distance:")
    print(f"EER: {standard_results['eer']:.4f}")
    print(f"EER Threshold: {standard_results['eer_threshold']:.4f}")
    
    print("\nModified Manhattan Distance:")
    print(f"EER: {modified_results['eer']:.4f}")
    print(f"EER Threshold: {modified_results['eer_threshold']:.4f}")
    
    # Calculate improvement
    improvement = (standard_results['eer'] - modified_results['eer']) / standard_results['eer'] * 100
    print(f"\nImprovement with modified metric: {improvement:.2f}%")
    
    # Plot results
    plot_results(standard_results, "Standard Manhattan Distance")
    plot_results(modified_results, "Modified Manhattan Distance")

    # Plot combined results for comparison
    plot_combined_results(standard_results, modified_results)
    
    return standard_results, modified_results
    


def plot_combined_results(standard_results: Dict, modified_results: Dict):
    """
    Plot both standard and modified Manhattan distance results on the same graph.
    
    Args:
        standard_results: Results from standard Manhattan distance
        modified_results: Results from modified Manhattan distance
    """
    plt.figure(figsize=(12, 8))
    
    # Plot standard Manhattan distance curves
    plt.plot(standard_results['thresholds'], standard_results['far'], 'b-', 
             label='Standard FAR', linewidth=2)
    plt.plot(standard_results['thresholds'], standard_results['frr'], 'b--', 
             label='Standard FRR', linewidth=2)
    
    # Plot modified Manhattan distance curves
    plt.plot(modified_results['thresholds'], modified_results['far'], 'r-', 
             label='Modified FAR', linewidth=2)
    plt.plot(modified_results['thresholds'], modified_results['frr'], 'r--', 
             label='Modified FRR', linewidth=2)
    
    # Plot EER points
    plt.scatter(standard_results['eer_threshold'], standard_results['eer'], 
                color='blue', s=100, marker='o',
                label=f'Standard EER: {standard_results["eer"]:.2%}')
    plt.scatter(modified_results['eer_threshold'], modified_results['eer'], 
                color='red', s=100, marker='o',
                label=f'Modified EER: {modified_results["eer"]:.2%}')
    
    # Add vertical lines at EER thresholds
    plt.axvline(x=standard_results['eer_threshold'], color='blue', linestyle=':', 
                alpha=0.7, label=f'Standard Threshold: {standard_results["eer_threshold"]:.2f}')
    plt.axvline(x=modified_results['eer_threshold'], color='red', linestyle=':', 
                alpha=0.7, label=f'Modified Threshold: {modified_results["eer_threshold"]:.2f}')
    
    # Set labels and title
    plt.xlabel('Distance Threshold', fontsize=12)
    plt.ylabel('Error Rate', fontsize=12)
    plt.title('Comparison of Standard vs. Modified Manhattan Distance', fontsize=14)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add a legend with good positioning
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    
    # Add a text box with the improvement percentage
    improvement = (standard_results['eer'] - modified_results['eer']) / standard_results['eer'] * 100
    plt.figtext(0.7, 0.15, f"Improvement: {improvement:.2f}%", 
                bbox=dict(facecolor='white', alpha=0.8), fontsize=12)
    
    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Save and show the plot
    plt.savefig("Manhattan_Distance_Comparison.png")
    plt.show()

def plot_results(results: Dict, title: str):
    """Plot FAR and FRR curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(results['thresholds'], results['far'], label='FAR')
    plt.plot(results['thresholds'], results['frr'], label='FRR')
    plt.axvline(x=results['eer_threshold'], color='r', linestyle='--', 
                label=f'EER Threshold: {results["eer_threshold"]:.2f}')
    plt.axhline(y=results['eer'], color='g', linestyle='--', 
                label=f'EER: {results["eer"]:.2%}')
    plt.xlabel('Threshold')
    plt.ylabel('Error Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

def evaluate_digraph_counts():
    """Evaluate the effect of different numbers of digraphs."""
    processor = KeystrokeProcessor()
    
    # Load all features
    if os.path.exists(os.path.join(processor.processed_data_dir, 'all_features.csv')):
        all_features = pd.read_csv(os.path.join(processor.processed_data_dir, 'all_features.csv'))
    else:
        print("Please run the main function first to process raw keystroke data.")
        return
    
    # Test different digraph counts
    digraph_counts = np.arange(1, 200, 5)
    eer_results = {'standard': [], 'modified': []}
    
    for digraph_count in digraph_counts:
        print(f"\nEvaluating with {digraph_count} digraphs...")
        extractor = KeystrokeFeatureExtractor(num_digraphs=digraph_count)
        extractor.identify_common_digraphs(all_features)
        feature_vectors = extractor.create_feature_vectors(all_features)
        
        # Apply appropriate normalizations for each method
        decimal_vectors = decimal_scaling(feature_vectors.copy())
        minmax_vectors = min_max_scaling(feature_vectors.copy())

        # Evaluate standard distance
        standard_authenticator = KeystrokeAuthenticator(du_total_weight=1.0)
        standard_eval = standard_authenticator.leave_one_out_evaluation(
            decimal_vectors, use_modified_distance=False)
        
        # Evaluate modified distance
        modified_authenticator = KeystrokeAuthenticator(du_total_weight=1/3)
        modified_eval = modified_authenticator.leave_one_out_evaluation(
            minmax_vectors, use_modified_distance=True)
        
        # Calculate EERs with appropriate threshold ranges for each method
        # For standard (decimal scaled)
        std_matrix = standard_eval['distance_matrix']
        std_valid_distances = std_matrix[np.isfinite(std_matrix) & (std_matrix > 0)]
        
        if len(std_valid_distances) == 0:
            print(f"  Skipping {digraph_count} digraphs: No valid distances found for standard.")
            eer_results['standard'].append(np.nan)
        else:
            std_min = np.min(std_valid_distances)
            std_max = np.max(std_valid_distances)
            
            if std_min >= std_max:
                print(f"  Standard: Min score ({std_min}) >= Max score ({std_max}). Using single threshold.")
                standard_thresholds = np.array([std_min])
            else:
                standard_thresholds = np.linspace(std_min, std_max, 1000)
            
            standard_results = standard_authenticator.calculate_error_rates(standard_eval, standard_thresholds)
            eer_results['standard'].append(standard_results['eer'])
            print(f"  Standard EER: {standard_results['eer']:.4f}, Threshold: {standard_results['eer_threshold']:.4f}")
        
        # For modified (min-max scaled)
        mod_matrix = modified_eval['distance_matrix']
        mod_valid_distances = mod_matrix[np.isfinite(mod_matrix) & (mod_matrix > 0)]
        
        if len(mod_valid_distances) == 0:
            print(f"  Skipping {digraph_count} digraphs: No valid distances found for modified.")
            eer_results['modified'].append(np.nan)
        else:
            mod_min = np.min(mod_valid_distances)
            mod_max = np.max(mod_valid_distances)
            
            if mod_min >= mod_max:
                print(f"  Modified: Min score ({mod_min}) >= Max score ({mod_max}). Using single threshold.")
                modified_thresholds = np.array([mod_min])
            else:
                modified_thresholds = np.linspace(mod_min, mod_max, 1000)
            
            modified_results = modified_authenticator.calculate_error_rates(modified_eval, modified_thresholds)
            eer_results['modified'].append(modified_results['eer'])
            print(f"  Modified EER: {modified_results['eer']:.4f}, Threshold: {modified_results['eer_threshold']:.4f}")
    
    # Plot results, handling potential NaNs
    valid_indices = ~(np.isnan(eer_results['standard']) | np.isnan(eer_results['modified']))
    valid_digraph_counts = [dc for i, dc in enumerate(digraph_counts) if valid_indices[i]]
    valid_standard = [eer for i, eer in enumerate(eer_results['standard']) if valid_indices[i]]
    valid_modified = [eer for i, eer in enumerate(eer_results['modified']) if valid_indices[i]]
    
    plt.figure(figsize=(10, 6))
    plt.plot(valid_digraph_counts, valid_standard, marker='o', label='Standard Manhattan')
    plt.plot(valid_digraph_counts, valid_modified, marker='s', label='Modified Manhattan')
    plt.xlabel('Number of Digraphs')
    plt.ylabel('EER')
    plt.title('Effect of Digraph Count on EER')
    plt.legend()
    plt.grid(True)
    plt.savefig("Digraph_Count_Effect.png")
    plt.show()

def main():
    # Process data and create feature vectors
    decimal_vectors, minmax_vectors = process_data()
    
    # Compare distance metrics using the appropriate normalization for each
    standard_results, modified_results = compare_distance_metrics(decimal_vectors, minmax_vectors)
    
    # Evaluate the effect of different digraph counts
    print("\nEvaluating the effect of different digraph counts...")
    evaluate_digraph_counts()

if __name__ == "__main__":
    main()