from detoxify import Detoxify
import pandas as pd
from pathlib import Path
import numpy as np
import os

class ContentFilter:
    def __init__(self, threshold=0.5):
        """Initialize the content filter with a toxicity threshold."""
        self.model = Detoxify('original')
        self.threshold = threshold
        
    def is_safe(self, text):
        """Check if a text is safe based on toxicity scores."""
        try:
            results = self.model.predict(text)
            # Check if any toxicity score exceeds the threshold
            return all(score < self.threshold for score in results.values())
        except Exception as e:
            print(f"Error checking content safety: {e}")
            return False
    
    def filter_dataset(self, df, safe_mode=True):
        """Filter the dataset based on safety scores."""
        if not safe_mode:
            return df
            
        print("Filtering dataset for safe content...")
        safe_cards = []
        
        for idx, row in df.iterrows():
            if self.is_safe(row['text']):
                safe_cards.append(row)
            if idx % 1000 == 0:
                print(f"Processed {idx} cards...")
                
        filtered_df = pd.DataFrame(safe_cards)
        print(f"Filtered dataset: {len(filtered_df)}/{len(df)} cards passed safety check")
        return filtered_df
    
    def get_toxicity_scores(self, text):
        """Get detailed toxicity scores for a text."""
        try:
            return self.model.predict(text)
        except Exception as e:
            print(f"Error getting toxicity scores: {e}")
            return None

def main():
    # Example usage
    filter = ContentFilter(threshold=0.5)
    
    # Get the current working directory
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # Load data
    data_dir = Path(current_dir) / 'data' / 'processed'
    print(f"Looking for data in: {data_dir}")
    
    # Check if directory exists
    if not data_dir.exists():
        print(f"Creating directory: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if the parquet file exists
    train_file = data_dir / 'cah_train.parquet'
    if not train_file.exists():
        print(f"Error: Training file not found at {train_file}")
        return
    
    # Load and process data
    print("Loading training data...")
    train_df = pd.read_parquet(train_file)
    
    # Filter dataset
    safe_df = filter.filter_dataset(train_df, safe_mode=True)
    
    # Save filtered dataset
    output_file = data_dir / 'cah_train_safe.parquet'
    print(f"Saving filtered dataset to: {output_file}")
    safe_df.to_parquet(output_file)
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Original size: {len(train_df)}")
    print(f"Safe mode size: {len(safe_df)}")
    print(f"Filtered out: {len(train_df) - len(safe_df)} cards")

if __name__ == "__main__":
    main() 