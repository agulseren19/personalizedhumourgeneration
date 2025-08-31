import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams as nltk_ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import os

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

def load_datasets():
    """Load both original and safe datasets."""
    data_dir = Path('data/processed')
    original_df = pd.read_parquet(data_dir / 'cah_train.parquet')
    safe_df = pd.read_parquet(data_dir / 'cah_train_safe.parquet')
    return original_df, safe_df

def compare_basic_stats(original_df, safe_df):
    """Compare basic statistics between datasets."""
    print("\n=== Basic Statistics Comparison ===")
    
    # Size comparison
    print("\nDataset Sizes:")
    print(f"Original dataset: {len(original_df)} cards")
    print(f"Safe dataset: {len(safe_df)} cards")
    print(f"Filtered out: {len(original_df) - len(safe_df)} cards")
    
    # Card type distribution
    print("\nCard Type Distribution:")
    print("Original dataset:")
    print(original_df['card_type'].value_counts(normalize=True))
    print("\nSafe dataset:")
    print(safe_df['card_type'].value_counts(normalize=True))
    
    # Length statistics
    print("\nLength Statistics:")
    for df, name in [(original_df, "Original"), (safe_df, "Safe")]:
        print(f"\n{name} dataset:")
        df['text_length'] = df['text'].str.len()
        print(df.groupby('card_type')['text_length'].describe())

def compare_word_frequencies(original_df, safe_df):
    """Compare word frequencies between datasets."""
    print("\n=== Word Frequency Comparison ===")
    
    def get_word_freq(texts):
        words = []
        for text in texts:
            try:
                # Clean and tokenize text
                text = str(text).lower()
                text = re.sub(r'[^\w\s]', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                tokens = text.split()
                # Filter out stopwords and short words
                words.extend([w for w in tokens if w not in stopwords.words('english') and len(w) > 1])
            except Exception as e:
                continue
        return Counter(words)
    
    # Compare black cards
    print("\nBlack Cards - Most Common Words:")
    print("Original dataset:")
    black_freq_orig = get_word_freq(original_df[original_df['card_type'] == 'black']['text'])
    print(black_freq_orig.most_common(10))
    print("\nSafe dataset:")
    black_freq_safe = get_word_freq(safe_df[safe_df['card_type'] == 'black']['text'])
    print(black_freq_safe.most_common(10))
    
    # Compare white cards
    print("\nWhite Cards - Most Common Words:")
    print("Original dataset:")
    white_freq_orig = get_word_freq(original_df[original_df['card_type'] == 'white']['text'])
    print(white_freq_orig.most_common(10))
    print("\nSafe dataset:")
    white_freq_safe = get_word_freq(safe_df[safe_df['card_type'] == 'white']['text'])
    print(white_freq_safe.most_common(10))

def compare_blank_patterns(original_df, safe_df):
    """Compare blank patterns between datasets."""
    print("\n=== Blank Pattern Comparison ===")
    
    for df, name in [(original_df, "Original"), (safe_df, "Safe")]:
        black_cards = df[df['card_type'] == 'black'].copy()
        black_cards['blank_count'] = black_cards['text'].str.count('_')
        
        print(f"\n{name} dataset blank distribution:")
        print(black_cards['blank_count'].value_counts().sort_index())

def compare_sentiment(original_df, safe_df):
    """Compare sentiment patterns between datasets."""
    print("\n=== Sentiment Analysis Comparison ===")
    
    for df, name in [(original_df, "Original"), (safe_df, "Safe")]:
        df['sentiment'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        print(f"\n{name} dataset sentiment statistics:")
        print(df.groupby('card_type')['sentiment'].describe())

def compare_lexical_diversity(original_df, safe_df):
    """Compare lexical diversity between datasets."""
    print("\n=== Lexical Diversity Comparison ===")
    
    def calculate_lexical_diversity(texts):
        all_words = []
        for text in texts:
            text = str(text).lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            words = text.split()
            all_words.extend(words)
        unique_words = set(all_words)
        return len(unique_words) / len(all_words) if all_words else 0
    
    for df, name in [(original_df, "Original"), (safe_df, "Safe")]:
        print(f"\n{name} dataset:")
        for card_type in ['black', 'white']:
            texts = df[df['card_type'] == card_type]['text']
            diversity = calculate_lexical_diversity(texts)
            print(f"{card_type.capitalize()} cards lexical diversity: {diversity:.3f}")

def compare_ngram_patterns(original_df, safe_df):
    """Compare n-gram patterns between datasets."""
    print("\n=== N-gram Pattern Comparison ===")
    
    def get_ngrams(texts, n):
        all_ngrams = []
        for text in texts:
            try:
                text = str(text).lower()
                text = re.sub(r'[^\w\s]', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                words = text.split()
                if len(words) >= n:
                    all_ngrams.extend(list(nltk_ngrams(words, n)))
            except Exception as e:
                continue
        return Counter(all_ngrams)
    
    for df, name in [(original_df, "Original"), (safe_df, "Safe")]:
        print(f"\n{name} dataset:")
        for card_type in ['black', 'white']:
            texts = df[df['card_type'] == card_type]['text']
            for n in [2, 3]:
                ngram_counts = get_ngrams(texts, n)
                print(f"\nTop 5 {n}-grams in {card_type} cards:")
                for ngram, count in ngram_counts.most_common(5):
                    print(f"  {' '.join(ngram)}: {count}")

def main():
    # Create analysis directory
    Path('analysis').mkdir(exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    original_df, safe_df = load_datasets()
    
    # Run comparisons
    compare_basic_stats(original_df, safe_df)
    compare_word_frequencies(original_df, safe_df)
    compare_blank_patterns(original_df, safe_df)
    compare_sentiment(original_df, safe_df)
    compare_lexical_diversity(original_df, safe_df)
    compare_ngram_patterns(original_df, safe_df)

if __name__ == "__main__":
    main() 