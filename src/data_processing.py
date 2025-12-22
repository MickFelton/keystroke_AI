import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set

class KeystrokeProcessor:
    """
    Process keystroke data from the free-text keystroke dynamics dataset.
    
    The data format is: key_code event_type timestamp
    where event_type is 0 for key press and 1 for key release.
    """
    
    def __init__(self, raw_data_dir='data/raw', processed_data_dir='data/processed'):
        """Initialize with paths to raw and processed data directories."""
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        
        # Create processed data directory if it doesn't exist
        if not os.path.exists(self.processed_data_dir):
            os.makedirs(self.processed_data_dir)
    
    def read_raw_file(self, user_id: str) -> pd.DataFrame:
        """Read raw keystroke data file for a specific user."""
        filepath = os.path.join(self.raw_data_dir, f'user{user_id}.txt')
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"User data file not found: {filepath}")
        
        # Read the data with proper column names
        df = pd.read_csv(filepath, sep=r'\s+', header=None, 
                         names=['key_code', 'event_type', 'timestamp'])
        
        # Add user_id column for identification
        df['user_id'] = user_id
        
        return df
    
    def process_all_users(self) -> pd.DataFrame:
        """
        Process data for all users in the raw data directory.
        
        Returns:
            DataFrame with digraph features for all users
        """
        # Get list of all user files
        user_files = [f for f in os.listdir(self.raw_data_dir) if f.startswith('user') and f.endswith('.txt')]
        
        all_features = []
        
        for user_file in user_files:
            user_id = user_file[4:-4]  # Extract user ID from filename (e.g., '0001' from 'user0001.txt')
            print(f"Processing data for user {user_id}...")
            
            # Read raw data
            df = self.read_raw_file(user_id)
            
            # Extract press and release events
            events = self.extract_press_release_events(df)
            
            # Segment events into chunks of ~1000 keystrokes
            segments = self.segment_events(events, segment_size=1000)
            
            # Process each segment
            for i, segment in enumerate(segments):
                segment_features = self.extract_digraph_features(segment)
                segment_features['segment_id'] = f"{user_id}_{i+1}"
                segment_features['user_id'] = user_id
                all_features.append(segment_features)
        
        # Combine all features into one DataFrame
        combined_features = pd.concat(all_features, ignore_index=True)
        
        # Save processed features
        output_file = os.path.join(self.processed_data_dir, 'all_features.csv')
        combined_features.to_csv(output_file, index=False)
        print(f"Saved all processed features to {output_file}")
        
        return combined_features
    
    def segment_events(self, events: pd.DataFrame, segment_size: int = 1000) -> List[pd.DataFrame]:
        """
        Split events dataframe into segments of approximately segment_size keys.
        
        Args:
            events: DataFrame with columns ['key', 'press_time', 'release_time']
            segment_size: Target number of keys per segment
            
        Returns:
            List of DataFrames, each containing a segment of events
        """
        total_keys = len(events)
        
        # If we have fewer than segment_size keys, return the entire dataset
        if total_keys <= segment_size:
            return [events]
        
        # Calculate number of segments
        num_segments = max(1, total_keys // segment_size)
        keys_per_segment = total_keys // num_segments
        
        segments = []
        for i in range(num_segments):
            start_idx = i * keys_per_segment
            end_idx = start_idx + keys_per_segment if i < num_segments - 1 else total_keys
            segment = events.iloc[start_idx:end_idx].copy()
            segments.append(segment)
        
        return segments
    
    def extract_press_release_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract press and release events from the raw data.
        
        Arguments:
            df: DataFrame with columns ['key_code', 'event_type', 'timestamp']
            
        Returns:
            DataFrame with event information (press and release timestamps for each key)
        """
        events = {
            'key': [],
            'press_time': [],
            'release_time': []
        }
        
        # In this dataset, 0 = press and 1 = release
        press_events = df[df['event_type'] == 0]
        release_events = df[df['event_type'] == 1]
        
        # Process each key press
        for _, press_row in press_events.iterrows():
            key = press_row['key_code']
            press_time = press_row['timestamp']
            
            # Find matching release event
            release_rows = release_events[
                (release_events['key_code'] == key) & 
                (release_events['timestamp'] > press_time)
            ]
            
            if not release_rows.empty:
                # Get the earliest release event after this press
                release_row = release_rows.iloc[0]
                release_time = release_row['timestamp']
                
                events['key'].append(key)
                events['press_time'].append(press_time)
                events['release_time'].append(release_time)
        
        # Convert to DataFrame for easier processing
        return pd.DataFrame(events)
    
    def extract_digraph_features(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Extract digraph features from the keystroke events.
        
        The features are:
        - DU1: Time from first key down to first key up
        - DU2: Time from second key down to second key up
        - DUtotal: Total time from first key down to second key up
        
        Arguments:
            events: DataFrame with columns ['key', 'press_time', 'release_time']
            
        Returns:
            DataFrame with digraph features
        """
        # Sort events by press time
        events = events.sort_values(by='press_time')
        
        digraph_features = {
            'first_key': [],
            'second_key': [],
            'digraph': [],
            'DU1': [],
            'DU2': [],
            'DUtotal': []
        }
        
        # We need at least 2 events to form a digraph
        if len(events) < 2:
            return pd.DataFrame(digraph_features)
        
        # Create digraphs
        for i in range(len(events) - 1):
            first_key = events.iloc[i]['key']
            second_key = events.iloc[i + 1]['key']
            
            # Calculate features
            first_key_down = events.iloc[i]['press_time']
            first_key_up = events.iloc[i]['release_time']
            second_key_down = events.iloc[i + 1]['press_time']
            second_key_up = events.iloc[i + 1]['release_time']
            
            # Ensure chronological order (sometimes release can be after next press in fast typing)
            if first_key_up > second_key_down:
                first_key_up = second_key_down
            
            du1 = first_key_up - first_key_down
            du2 = second_key_up - second_key_down
            du_total = second_key_up - first_key_down
            
            digraph_features['first_key'].append(first_key)
            digraph_features['second_key'].append(second_key)
            digraph_features['digraph'].append(f"{first_key}-{second_key}")
            digraph_features['DU1'].append(du1)
            digraph_features['DU2'].append(du2)
            digraph_features['DUtotal'].append(du_total)
        
        return pd.DataFrame(digraph_features)