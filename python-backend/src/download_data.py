import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import requests
from tqdm import tqdm

def download_file(url, output_path):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def process_cah_data(input_path, output_dir):
    """Process CAH data into train/val/test splits."""
    print("Reading JSON file...")
    # Read the JSON data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("Processing cards...")
    # Extract black and white cards
    rows = []
    
    # Process white cards
    for text in tqdm(data['white'], desc="Processing white cards"):
        rows.append({
            'card_type': 'white',
            'text': text.strip(),
            'pack': 'all'  # We'll add pack info later if needed
        })
    
    # Process black cards
    for card in tqdm(data['black'], desc="Processing black cards"):
        if isinstance(card, dict) and 'text' in card:
            rows.append({
                'card_type': 'black',
                'text': card['text'].strip(),
                'pick': card.get('pick', 1),
                'pack': 'all'  # We'll add pack info later if needed
            })
    
    print("Creating DataFrame...")
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['text', 'card_type'])
    
    print("Splitting data...")
    # Split into train/val/test
    train, test = train_test_split(
        df, test_size=0.1, random_state=42, stratify=df['card_type'])
    train, valid = train_test_split(
        train, test_size=0.05, random_state=42, stratify=train['card_type'])
    
    # Save splits
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Saving processed data...")
    train.to_parquet(output_dir / 'cah_train.parquet')
    valid.to_parquet(output_dir / 'cah_valid.parquet')
    test.to_parquet(output_dir / 'cah_test.parquet')
    
    print("\nProcessing complete!")
    print(f"Processed {len(df)} unique cards")
    print(f"Black cards: {len(df[df['card_type'] == 'black'])}")
    print(f"White cards: {len(df[df['card_type'] == 'white'])}")
    print(f"Train: {len(train)}, Validation: {len(valid)}, Test: {len(test)}")
    
    # Print some example cards
    print("\nExample black cards:")
    print(df[df['card_type'] == 'black']['text'].head().to_string())
    print("\nExample white cards:")
    print(df[df['card_type'] == 'white']['text'].head().to_string())

def main():
    # Create data directories
    data_dir = Path('data')
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Download CAH dataset
    cah_url = "https://raw.githubusercontent.com/crhallberg/json-against-humanity/master/cah-all-compact.json"
    cah_path = raw_dir / "cah-all-compact.json"
    
    if not cah_path.exists():
        print("Downloading CAH dataset...")
        download_file(cah_url, cah_path)
    
    # Process the data
    print("Processing CAH data...")
    process_cah_data(cah_path, processed_dir)

if __name__ == "__main__":
    main() 