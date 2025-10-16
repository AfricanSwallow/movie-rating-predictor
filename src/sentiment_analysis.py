"""
Sentiment Analysis Module for Movie Review Analysis

This module provides functions for sentiment analysis using:
- TextBlob for simple polarity-based sentiment
- Custom lexicon-based approach
"""

from typing import Dict, List, Tuple
import pandas as pd


def get_sentiment_lexicon() -> Tuple[set, set]:
    """
    Get positive and negative word lexicons for sentiment analysis.
    
    Returns:
        Tuple of (positive_words, negative_words) sets
    """
    # Common positive words in movie reviews
    positive_words = {
        'amazing', 'awesome', 'beautiful', 'best', 'better', 'brilliant', 'classic', 
        'excellent', 'exceptional', 'exciting', 'fantastic', 'favorite', 'fun', 'funny',
        'good', 'great', 'incredible', 'interesting', 'love', 'loved', 'lovely', 
        'masterpiece', 'outstanding', 'perfect', 'powerful', 'recommend', 'superb',
        'terrific', 'wonderful', 'enjoy', 'enjoyed', 'entertaining', 'impressive',
        'spectacular', 'stunning', 'touching', 'compelling', 'captivating', 'genius',
        'brilliant', 'innovative', 'unique', 'original', 'refreshing', 'delightful',
        'pleasant', 'charming', 'satisfying', 'rewarding', 'worthwhile', 'must-see',
        'hilarious', 'thrilling', 'gripping', 'intense', 'moving', 'heartwarming',
        'beautiful', 'gorgeous', 'amazing', 'incredible', 'phenomenal', 'extraordinary'
    }
    
    # Common negative words in movie reviews
    negative_words = {
        'awful', 'bad', 'boring', 'terrible', 'horrible', 'worst', 'worse', 'poor',
        'disappointing', 'disappointed', 'dull', 'waste', 'wasted', 'predictable',
        'cliche', 'mediocre', 'weak', 'lacks', 'lacking', 'fail', 'fails', 'failed',
        'failure', 'pointless', 'stupid', 'ridiculous', 'pathetic', 'lame', 'poor',
        'annoying', 'irritating', 'frustrating', 'confusing', 'confused', 'mess',
        'messy', 'disaster', 'unwatchable', 'unbearable', 'cringe', 'crappy',
        'garbage', 'trash', 'rubbish', 'skip', 'avoid', 'regret', 'unfortunately',
        'sadly', 'hate', 'hated', 'dislike', 'disliked', 'awful', 'atrocious',
        'abysmal', 'appalling', 'dreadful', 'miserable', 'unpleasant', 'uninteresting'
    }
    
    return positive_words, negative_words


def lexicon_based_sentiment(tokens: List[str], 
                            positive_words: set = None,
                            negative_words: set = None) -> Dict[str, float]:
    """
    Calculate sentiment based on positive/negative word counts in tokens.
    
    Args:
        tokens: List of tokens (words)
        positive_words: Set of positive sentiment words
        negative_words: Set of negative sentiment words
        
    Returns:
        Dictionary with sentiment scores
    """
    if not tokens:
        return {
            'positive_count': 0,
            'negative_count': 0,
            'sentiment_score': 0.0,
            'sentiment_label': 'neutral'
        }
    
    # Get default lexicons if not provided
    if positive_words is None or negative_words is None:
        positive_words, negative_words = get_sentiment_lexicon()
    
    # Count positive and negative words
    pos_count = sum(1 for token in tokens if token in positive_words)
    neg_count = sum(1 for token in tokens if token in negative_words)
    
    # Calculate sentiment score (-1 to 1)
    total_sentiment_words = pos_count + neg_count
    if total_sentiment_words == 0:
        sentiment_score = 0.0
    else:
        sentiment_score = (pos_count - neg_count) / len(tokens)
    
    # Determine sentiment label
    if sentiment_score > 0.02:
        sentiment_label = 'positive'
    elif sentiment_score < -0.02:
        sentiment_label = 'negative'
    else:
        sentiment_label = 'neutral'
    
    return {
        'positive_count': pos_count,
        'negative_count': neg_count,
        'sentiment_score': sentiment_score,
        'sentiment_label': sentiment_label
    }


def textblob_sentiment(text: str) -> Dict[str, float]:
    """
    Calculate sentiment using TextBlob library.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with polarity and subjectivity scores
    """
    try:
        from textblob import TextBlob
    except ImportError:
        print("TextBlob not installed. Install with: pip install textblob")
        return {
            'polarity': 0.0,
            'subjectivity': 0.0,
            'sentiment_label': 'neutral'
        }
    
    if pd.isna(text) or text == "":
        return {
            'polarity': 0.0,
            'subjectivity': 0.0,
            'sentiment_label': 'neutral'
        }
    
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Determine sentiment label
    if polarity > 0.1:
        sentiment_label = 'positive'
    elif polarity < -0.1:
        sentiment_label = 'negative'
    else:
        sentiment_label = 'neutral'
    
    return {
        'polarity': polarity,
        'subjectivity': subjectivity,
        'sentiment_label': sentiment_label
    }


def analyze_dataframe_sentiment(df: pd.DataFrame,
                                text_column: str = None,
                                token_column: str = None,
                                method: str = 'lexicon') -> pd.DataFrame:
    """
    Apply sentiment analysis to a DataFrame.
    
    Args:
        df: Input DataFrame
        text_column: Column containing text (for TextBlob method)
        token_column: Column containing tokens (for lexicon method)
        method: 'lexicon' or 'textblob'
        
    Returns:
        DataFrame with sentiment columns added
    """
    print(f"Performing sentiment analysis using {method} method...")
    
    if method == 'lexicon':
        if token_column is None:
            raise ValueError("token_column must be specified for lexicon method")
        
        # Apply lexicon-based sentiment
        sentiment_results = df[token_column].apply(lexicon_based_sentiment)
        
        # Extract results into separate columns
        df['positive_count'] = sentiment_results.apply(lambda x: x['positive_count'])
        df['negative_count'] = sentiment_results.apply(lambda x: x['negative_count'])
        df['sentiment_score'] = sentiment_results.apply(lambda x: x['sentiment_score'])
        df['sentiment_label'] = sentiment_results.apply(lambda x: x['sentiment_label'])
        
    elif method == 'textblob':
        if text_column is None:
            raise ValueError("text_column must be specified for textblob method")
        
        # Apply TextBlob sentiment
        sentiment_results = df[text_column].apply(textblob_sentiment)
        
        # Extract results into separate columns
        df['polarity'] = sentiment_results.apply(lambda x: x['polarity'])
        df['subjectivity'] = sentiment_results.apply(lambda x: x['subjectivity'])
        df['sentiment_label'] = sentiment_results.apply(lambda x: x['sentiment_label'])
        
    else:
        raise ValueError("method must be 'lexicon' or 'textblob'")
    
    print(f"✓ Sentiment analysis complete!")
    print(f"\nSentiment distribution:")
    print(df['sentiment_label'].value_counts())
    
    return df


def get_sentiment_statistics(df: pd.DataFrame) -> None:
    """
    Print sentiment statistics.
    
    Args:
        df: DataFrame with sentiment columns
    """
    print("=" * 80)
    print("SENTIMENT ANALYSIS STATISTICS")
    print("=" * 80)
    
    # Overall distribution
    print("\n1. Sentiment Label Distribution:")
    sentiment_counts = df['sentiment_label'].value_counts()
    sentiment_pct = df['sentiment_label'].value_counts(normalize=True) * 100
    
    for label in ['positive', 'neutral', 'negative']:
        if label in sentiment_counts.index:
            count = sentiment_counts[label]
            pct = sentiment_pct[label]
            print(f"   {label.capitalize()}: {count:,} ({pct:.1f}%)")
    
    # Score statistics
    if 'sentiment_score' in df.columns:
        print("\n2. Sentiment Score Statistics:")
        print(f"   Mean: {df['sentiment_score'].mean():.4f}")
        print(f"   Median: {df['sentiment_score'].median():.4f}")
        print(f"   Std: {df['sentiment_score'].std():.4f}")
        print(f"   Min: {df['sentiment_score'].min():.4f}")
        print(f"   Max: {df['sentiment_score'].max():.4f}")
        
        if 'positive_count' in df.columns and 'negative_count' in df.columns:
            print("\n3. Sentiment Word Counts:")
            print(f"   Avg positive words per review: {df['positive_count'].mean():.2f}")
            print(f"   Avg negative words per review: {df['negative_count'].mean():.2f}")
    
    elif 'polarity' in df.columns:
        print("\n2. Polarity Statistics:")
        print(f"   Mean: {df['polarity'].mean():.4f}")
        print(f"   Median: {df['polarity'].median():.4f}")
        print(f"   Std: {df['polarity'].std():.4f}")
        
        print("\n3. Subjectivity Statistics:")
        print(f"   Mean: {df['subjectivity'].mean():.4f}")
        print(f"   Median: {df['subjectivity'].median():.4f}")
    
    print("=" * 80)


if __name__ == "__main__":
    # Test sentiment analysis
    sample_reviews = [
        "This movie was absolutely amazing! The acting was brilliant and the story was incredible.",
        "Terrible waste of time. The plot was confusing and the characters were boring.",
        "It was okay, not great but not terrible either. Some parts were interesting.",
        "I loved every minute of this film! Masterpiece of cinema, highly recommend!",
        "Worst movie ever. Boring, predictable, and poorly made. Total disaster."
    ]
    
    print("Sentiment Analysis Examples:")
    print("=" * 80)
    
    # Test lexicon-based (need tokens)
    from text_tokenization import tokenize_and_process
    
    for i, review in enumerate(sample_reviews, 1):
        tokens = tokenize_and_process(review)
        sentiment = lexicon_based_sentiment(tokens)
        
        print(f"\nReview {i}: {review[:60]}...")
        print(f"Tokens: {tokens[:10]}")
        print(f"Sentiment: {sentiment['sentiment_label']} (score: {sentiment['sentiment_score']:.3f})")
        print(f"Positive words: {sentiment['positive_count']}, Negative words: {sentiment['negative_count']}")
