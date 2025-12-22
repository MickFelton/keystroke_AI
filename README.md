# Keystroke Dynamics Authentication System

**Course Project:** CSN-371 Artificial Intelligence  
**Author:** Mohammed Haaziq Jamal  
**Instructor:** Prof. Pradumn K. Pandey  

A biometric authentication system that identifies users based on their unique typing patterns using keystroke dynamics. This implementation uses digraph timing features and evaluates performance using the Equal Error Rate (EER) metric, based on research by Iapa & Cretu (2021).

## 📋 Overview

This project implements a keystroke dynamics authentication system that:
- Extracts timing features from keystroke data (digraphs)
- Uses modified Manhattan distance metric for improved accuracy
- Evaluates authentication performance using leave-one-out methodology
- Compares standard and modified distance metrics
- Analyzes the effect of feature selection on authentication accuracy

## 🎯 Key Features

- **Digraph Feature Extraction**: Analyzes timing patterns between consecutive keystrokes
  - DU1: First key down to first key up (dwell time)
  - DU2: Second key down to second key up (dwell time)
  - DUtotal: First key down to second key up (total time)

- **Modified Manhattan Distance**: Implements weighted distance metric with reduced weight for DUtotal features (default: 1/3)

- **Multiple Normalization Techniques**:
  - Decimal scaling for standard Manhattan distance
  - Min-max scaling for modified Manhattan distance

- **Comprehensive Evaluation**:
  - Leave-one-out cross-validation
  - FAR (False Accept Rate) and FRR (False Reject Rate) calculations
  - EER (Equal Error Rate) computation
  - Visualization of error rates vs. thresholds

## 📁 Project Structure

```
keystroke_AI/
├── data/
│   ├── raw/                    # Raw keystroke data files (user0001.txt - user0080.txt)
│   └── processed/              # Processed feature vectors
│       ├── all_features.csv
│       ├── decimal_vectors.csv
│       └── minmax_vectors.csv
├── src/
│   ├── data_processing.py      # Data loading and digraph extraction
│   ├── feature_extraction.py   # Feature vector creation
│   ├── metrics.py              # Distance metric implementations
│   └── authentication.py       # Authentication logic and evaluation
├── main.py                     # Main execution script
└── README.md                   # This file
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Haaziq386/keystroke_AI.git
cd keystroke_AI
```

2. Ensure your data is in the correct format in `data/raw/`:
   - Files named `user####.txt` (e.g., `user0001.txt`)
   - Format: `key_code event_type timestamp`
     - `key_code`: ASCII code of the key
     - `event_type`: 0 for press, 1 for release
     - `timestamp`: Milliseconds since epoch

### Usage

Run the main script to process data and evaluate authentication:

```bash
python main.py
```

This will:
1. Process raw keystroke data files
2. Extract digraph features
3. Create and normalize feature vectors
4. Compare standard vs. modified Manhattan distance metrics
5. Evaluate the effect of different numbers of digraphs
6. Generate visualization plots

## 📊 Data Format

### Raw Data Format
Each user file contains keystroke events in the format:
```
key_code event_type timestamp
16 0 434889        # Key 16 pressed at time 434889
86 0 435006        # Key 86 pressed at time 435006
86 1 435146        # Key 86 released at time 435146
16 1 435221        # Key 16 released at time 435221
```

### Feature Vector Format
Each segment is represented by timing features for the most common digraphs:
```
segment_id, user_id, [digraph]_DU1, [digraph]_DU2, [digraph]_DUtotal, ...
```

## 🔬 Methodology

### 1. Data Processing
- Load raw keystroke data
- Match press and release events
- Segment into chunks of ~1000 keystrokes

### 2. Feature Extraction
- Identify the most common digraphs (default: 12)
- Calculate DU1, DU2, and DUtotal for each digraph
- Create feature vectors using median values per segment

### 3. Normalization
- **Decimal Scaling**: For standard Manhattan distance
- **Min-Max Scaling**: For modified Manhattan distance

### 4. Authentication
- Leave-one-out cross-validation
- Distance calculation between feature vectors
- Threshold-based classification
- FAR/FRR/EER computation

## 📈 Performance Metrics

- **FAR (False Accept Rate)**: Percentage of impostor attempts incorrectly accepted
- **FRR (False Reject Rate)**: Percentage of genuine attempts incorrectly rejected
- **EER (Equal Error Rate)**: Point where FAR equals FRR (lower is better)

## 🎨 Visualizations

The system generates several plots:
- `Standard_Manhattan_Distance.png`: FAR/FRR curves for standard metric
- `Modified_Manhattan_Distance.png`: FAR/FRR curves for modified metric
- `Manhattan_Distance_Comparison.png`: Side-by-side comparison
- `Digraph_Count_Effect.png`: EER vs. number of digraphs

## 🔧 Configuration

### Adjustable Parameters

In `main.py`:
- `segment_size`: Number of keystrokes per segment (default: 1000)
- `num_digraphs`: Number of most common digraphs to use (default: 12)

In `authentication.py`:
- `du_total_weight`: Weight for DUtotal in modified distance (default: 1/3)

## 📚 Module Reference

### `KeystrokeProcessor`
Handles raw data loading and preprocessing:
- `read_raw_file(user_id)`: Load data for a specific user
- `process_all_users()`: Process all users and extract digraphs
- `extract_digraph_features(events)`: Calculate timing features

### `KeystrokeFeatureExtractor`
Creates feature vectors:
- `identify_common_digraphs(all_features)`: Find most frequent digraphs
- `create_feature_vectors(all_features)`: Build feature vectors

### `KeystrokeMetrics`
Implements distance metrics:
- `manhattan_distance(v1, v2)`: Standard Manhattan distance
- `modified_manhattan_distance(v1, v2)`: Weighted Manhattan distance

### `KeystrokeAuthenticator`
Performs authentication and evaluation:
- `leave_one_out_evaluation(feature_vectors)`: Cross-validation
- `calculate_error_rates(evaluation_results, thresholds)`: Compute FAR/FRR/EER

## 🧪 Experimental Results

The modified Manhattan distance metric with reduced DUtotal weight typically achieves:
- Lower EER compared to standard Manhattan distance
- Better discrimination between genuine and impostor attempts
- Optimal performance with ~12 most common digraphs

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

Haaziq - [GitHub](https://github.com/Haaziq386)

## 🙏 Acknowledgments

This implementation is based on research in keystroke dynamics authentication, particularly the use of digraph timing features and modified distance metrics for improved accuracy.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.
