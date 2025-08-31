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
from nltk.tokenize import word_tokenize
from nltk.util import ngrams as nltk_ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import networkx as nx
from scipy.stats import entropy

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def load_data():
    """Load the processed dataset."""
    data_dir = Path('data/processed')
    train_df = pd.read_parquet(data_dir / 'cah_train.parquet')
    valid_df = pd.read_parquet(data_dir / 'cah_valid.parquet')
    test_df = pd.read_parquet(data_dir / 'cah_test.parquet')
    return train_df, valid_df, test_df

def analyze_card_lengths(df):
    """Analyze the length distribution of cards."""
    df['text_length'] = df['text'].str.len()
    
    # Plot length distributions
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='text_length', hue='card_type', bins=50)
    plt.title('Distribution of Card Lengths by Type')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Count')
    plt.savefig('analysis/card_lengths.png')
    plt.close()
    
    # Print statistics
    print("\nCard Length Statistics:")
    print(df.groupby('card_type')['text_length'].describe())

def clean_text(text):
    """Clean and tokenize text."""
    # Convert to string and lowercase
    text = str(text).lower()
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Split into words
    return text.split()

def analyze_word_frequencies(df):
    """Analyze word frequencies in cards."""
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    def get_word_freq(texts):
        words = []
        for text in texts:
            try:
                # Clean and tokenize text
                tokens = clean_text(text)
                # Filter out stopwords and short words
                words.extend([w for w in tokens if w not in stop_words and len(w) > 1])
            except Exception as e:
                print(f"Error processing text: {text}")
                print(f"Error: {e}")
                continue
        return Counter(words)
    
    # Get word frequencies for each card type
    print("\nProcessing black cards...")
    black_freq = get_word_freq(df[df['card_type'] == 'black']['text'])
    print("Processing white cards...")
    white_freq = get_word_freq(df[df['card_type'] == 'white']['text'])
    
    # Create word clouds
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(black_freq)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Black Cards Word Cloud')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(white_freq)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('White Cards Word Cloud')
    plt.axis('off')
    
    plt.savefig('analysis/word_clouds.png')
    plt.close()
    
    # Print most common words
    print("\nMost Common Words in Black Cards:")
    print(black_freq.most_common(20))
    print("\nMost Common Words in White Cards:")
    print(white_freq.most_common(20))

def analyze_blank_patterns(df):
    """Analyze patterns in black cards with blanks."""
    black_cards = df[df['card_type'] == 'black'].copy()  # Create a copy to avoid SettingWithCopyWarning
    
    # Count number of blanks per card
    black_cards['blank_count'] = black_cards['text'].str.count('_')
    
    # Plot blank count distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=black_cards, x='blank_count')
    plt.title('Distribution of Number of Blanks in Black Cards')
    plt.xlabel('Number of Blanks')
    plt.ylabel('Count')
    plt.savefig('analysis/blank_counts.png')
    plt.close()
    
    # Print statistics
    print("\nBlank Count Statistics:")
    print(black_cards['blank_count'].value_counts().sort_index())

def analyze_lexical_diversity(df):
    """Analyze lexical diversity and vocabulary richness."""
    def calculate_lexical_diversity(texts):
        all_words = []
        for text in texts:
            words = clean_text(text)
            all_words.extend(words)
        unique_words = set(all_words)
        return len(unique_words) / len(all_words) if all_words else 0
    
    print("\nLexical Diversity Analysis:")
    for card_type in ['black', 'white']:
        texts = df[df['card_type'] == card_type]['text']
        diversity = calculate_lexical_diversity(texts)
        print(f"{card_type.capitalize()} cards lexical diversity: {diversity:.3f}")

def analyze_sentiment(df):
    """Analyze sentiment patterns in cards."""
    def get_sentiment(text):
        return TextBlob(str(text)).sentiment.polarity
    
    df['sentiment'] = df['text'].apply(get_sentiment)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='sentiment', hue='card_type', bins=50)
    plt.title('Sentiment Distribution by Card Type')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Count')
    plt.savefig('analysis/sentiment_distribution.png')
    plt.close()
    
    print("\nSentiment Analysis:")
    print(df.groupby('card_type')['sentiment'].describe())

def analyze_ngram_patterns(df):
    """Analyze n-gram patterns in cards."""
    def get_ngrams(texts, n):
        all_ngrams = []
        for text in texts:
            try:
                words = clean_text(text)
                if len(words) >= n:  # Only process if we have enough words
                    all_ngrams.extend(list(nltk_ngrams(words, n)))
            except Exception as e:
                print(f"Error processing text for n-grams: {text}")
                print(f"Error: {e}")
                continue
        return Counter(all_ngrams)
    
    print("\nN-gram Analysis:")
    for card_type in ['black', 'white']:
        texts = df[df['card_type'] == card_type]['text']
        for n in [2, 3]:
            try:
                ngram_counts = get_ngrams(texts, n)
                print(f"\nTop 5 {n}-grams in {card_type} cards:")
                for ngram, count in ngram_counts.most_common(5):
                    print(f"  {' '.join(ngram)}: {count}")
            except Exception as e:
                print(f"Error analyzing {n}-grams for {card_type} cards: {e}")

def analyze_surprise_index(df):
    """Calculate surprise index based on word probabilities."""
    try:
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        texts = df['text'].fillna('')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Calculate average TF-IDF scores
        avg_scores = tfidf_matrix.mean(axis=0).A1
        word_scores = dict(zip(vectorizer.get_feature_names_out(), avg_scores))
        
        # Calculate surprise index for each card type
        print("\nSurprise Index Analysis:")
        for card_type in ['black', 'white']:
            card_texts = df[df['card_type'] == card_type]['text']
            surprise_scores = []
            for text in card_texts:
                words = clean_text(text)
                if words:
                    score = np.mean([word_scores.get(w, 0) for w in words])
                    surprise_scores.append(score)
            print(f"{card_type.capitalize()} cards average surprise index: {np.mean(surprise_scores):.3f}")
    except Exception as e:
        print(f"Error in surprise index analysis: {e}")

def analyze_card_combinations():
    """Analyze potential card combinations."""
    try:
        train_df, _, _ = load_data()
        
        # Get sample of black and white cards
        black_cards = train_df[train_df['card_type'] == 'black'].sample(5)
        white_cards = train_df[train_df['card_type'] == 'white'].sample(5)
        
        print("\nSample Card Combinations:")
        for _, black in black_cards.iterrows():
            print(f"\nBlack Card: {black['text']}")
            for _, white in white_cards.iterrows():
                combination = black['text'].replace('_', white['text'])
                print(f"  + {white['text']} = {combination}")
    except Exception as e:
        print(f"Error in card combinations analysis: {e}")

def main():
    try:
        # Create analysis directory
        Path('analysis').mkdir(exist_ok=True)
        
        # Load data
        train_df, valid_df, test_df = load_data()
        
        print("Dataset Statistics:")
        print(f"Total cards: {len(train_df) + len(valid_df) + len(test_df)}")
        print(f"Black cards: {len(train_df[train_df['card_type'] == 'black'])}")
        print(f"White cards: {len(train_df[train_df['card_type'] == 'white'])}")
        
        # Run analyses
        analyze_card_lengths(train_df)
        analyze_word_frequencies(train_df)
        analyze_blank_patterns(train_df)
        analyze_lexical_diversity(train_df)
        analyze_sentiment(train_df)
        analyze_ngram_patterns(train_df)
        analyze_surprise_index(train_df)
        analyze_card_combinations()
    except Exception as e:
        print(f"Error in main analysis: {e}")

if __name__ == "__main__":
    main() 