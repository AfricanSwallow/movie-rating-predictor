"""
Text Tokenization and Processing Module for Movie Review Analysis

This module provides functions for:
- Tokenization (breaking text into words)
- Stop word removal
- Stemming/Lemmatization
- Token filtering and processing
"""

import re
from typing import List, Optional, Set
import pandas as pd


def simple_tokenize(text: str) -> List[str]:
    """
    Simple tokenization by splitting on whitespace.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of tokens
    """
    if pd.isna(text) or text == "":
        return []
    
    return text.split()


def get_english_stopwords() -> Set[str]:
    """
    Get a comprehensive set of English stop words.
    
    Returns:
        Set of stop words
    """
    # Common English stop words
    stopwords = {
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 
        'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 
        'both', 'but', 'by', 'can', 'cannot', 'could', 'did', 'do', 'does', 'doing', 'down', 
        'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having', 
        'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 
        'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'might', 'more', 'most', 
        'must', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'only', 
        'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 's', 'same', 'she', 
        'should', 'so', 'some', 'such', 't', 'than', 'that', 'the', 'their', 'theirs', 'them', 
        'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 
        'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 
        'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'you', 'your', 'yours', 
        'yourself', 'yourselves'
    }
    
    return stopwords


def remove_stopwords(tokens: List[str], custom_stopwords: Optional[Set[str]] = None) -> List[str]:
    """
    Remove stop words from token list.
    
    Args:
        tokens: List of tokens
        custom_stopwords: Optional set of custom stop words to use instead of default
        
    Returns:
        List of tokens with stop words removed
    """
    if not tokens:
        return []
    
    if custom_stopwords is None:
        stopwords = get_english_stopwords()
    else:
        stopwords = custom_stopwords
    
    return [token for token in tokens if token.lower() not in stopwords]


def filter_tokens(tokens: List[str], 
                 min_length: int = 2,
                 max_length: Optional[int] = None,
                 remove_numbers: bool = True) -> List[str]:
    """
    Filter tokens based on various criteria.
    
    Args:
        tokens: List of tokens
        min_length: Minimum token length to keep
        max_length: Maximum token length to keep (None = no limit)
        remove_numbers: Remove tokens that are purely numeric
        
    Returns:
        Filtered list of tokens
    """
    if not tokens:
        return []
    
    filtered = []
    for token in tokens:
        # Check length
        if len(token) < min_length:
            continue
        if max_length is not None and len(token) > max_length:
            continue
        
        # Check if purely numeric
        if remove_numbers and token.isdigit():
            continue
        
        filtered.append(token)
    
    return filtered


def simple_stem(word: str) -> str:
    """
    Simple suffix stripping stemmer (removes common suffixes).
    This is a basic implementation - for production use, consider NLTK's PorterStemmer.
    
    Args:
        word: Word to stem
        
    Returns:
        Stemmed word
    """
    if len(word) <= 3:
        return word
    
    # Common suffix patterns
    suffixes = [
        ('sses', 'ss'),  #asses -> ass
        ('ies', 'i'),    # cries -> cri
        ('ss', 'ss'),    # stress -> stress
        ('s', ''),       # cats -> cat
        ('ing', ''),     # running -> runn
        ('ed', ''),      # played -> play
        ('ly', ''),      # quickly -> quick
        ('er', ''),      # faster -> fast
        ('est', ''),     # fastest -> fast
    ]
    
    word_lower = word.lower()
    for suffix, replacement in suffixes:
        if word_lower.endswith(suffix):
            return word_lower[:len(word_lower)-len(suffix)] + replacement
    
    return word_lower


def stem_tokens(tokens: List[str]) -> List[str]:
    """
    Apply stemming to a list of tokens.
    
    Args:
        tokens: List of tokens
        
    Returns:
        List of stemmed tokens
    """
    if not tokens:
        return []
    
    return [simple_stem(token) for token in tokens]


def tokenize_and_process(text: str,
                        remove_stopwords_flag: bool = True,
                        apply_stemming: bool = True,
                        min_token_length: int = 2,
                        max_token_length: Optional[int] = None,
                        remove_numbers: bool = True,
                        custom_stopwords: Optional[Set[str]] = None) -> List[str]:
    """
    Complete tokenization and processing pipeline.
    
    Args:
        text: Input text to process
        remove_stopwords_flag: Remove stop words
        apply_stemming: Apply stemming to tokens
        min_token_length: Minimum token length
        max_token_length: Maximum token length
        remove_numbers: Remove numeric tokens
        custom_stopwords: Custom stop word set
        
    Returns:
        List of processed tokens
    """
    if pd.isna(text) or text == "":
        return []
    
    # Tokenize
    tokens = simple_tokenize(text)
    
    # Filter tokens
    tokens = filter_tokens(tokens, min_token_length, max_token_length, remove_numbers)
    
    # Remove stop words
    if remove_stopwords_flag:
        tokens = remove_stopwords(tokens, custom_stopwords)
    
    # Apply stemming
    if apply_stemming:
        tokens = stem_tokens(tokens)
    
    return tokens


def process_dataframe_column(df: pd.DataFrame,
                            column_name: str,
                            output_column: Optional[str] = None,
                            join_tokens: bool = False,
                            **kwargs) -> pd.DataFrame:
    """
    Apply tokenization and processing to a DataFrame column.
    
    Args:
        df: Input DataFrame
        column_name: Name of column to process
        output_column: Name for output column (default: column_name + '_tokens')
        join_tokens: If True, join tokens back into string; if False, keep as list
        **kwargs: Additional arguments to pass to tokenize_and_process()
        
    Returns:
        DataFrame with processed tokens column added
    """
    if output_column is None:
        output_column = f"{column_name}_tokens"
    
    print(f"Processing tokens in column '{column_name}'...")
    df[output_column] = df[column_name].apply(lambda x: tokenize_and_process(x, **kwargs))
    
    if join_tokens:
        df[output_column] = df[output_column].apply(lambda x: ' '.join(x) if x else '')
    
    print(f"✓ Processed tokens saved to column '{output_column}'")
    
    return df


def get_token_statistics(df: pd.DataFrame, token_column: str) -> pd.DataFrame:
    """
    Get statistics about tokens in a DataFrame column.
    
    Args:
        df: DataFrame with token column
        token_column: Name of column containing token lists
        
    Returns:
        DataFrame with token statistics
    """
    # Count tokens per review
    df['_token_count'] = df[token_column].apply(len)
    
    print("Token Statistics:")
    print(f"Total reviews: {len(df)}")
    print(f"Average tokens per review: {df['_token_count'].mean():.1f}")
    print(f"Median tokens per review: {df['_token_count'].median():.1f}")
    print(f"Min tokens: {df['_token_count'].min()}")
    print(f"Max tokens: {df['_token_count'].max()}")
    
    # Get all unique tokens
    from collections import Counter
    all_tokens = []
    for tokens in df[token_column]:
        all_tokens.extend(tokens)
    
    token_freq = Counter(all_tokens)
    print(f"\nUnique tokens: {len(token_freq)}")
    print(f"Total tokens: {len(all_tokens)}")
    
    # Top 20 most common tokens
    print("\nTop 20 most common tokens:")
    for token, count in token_freq.most_common(20):
        print(f"  {token}: {count}")
    
    # Clean up temporary column
    df.drop('_token_count', axis=1, inplace=True)
    
    return df


if __name__ == "__main__":
    # Test the tokenization functions
    sample_texts = [
        "this is a great movie with amazing acting",
        "i cannot believe how wonderful this film was it is the best",
        "the special effects were incredible and the story was compelling",
    ]
    
    print("Tokenization Examples:")
    print("=" * 80)
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\nText {i}: {text}")
        
        # Basic tokenization
        tokens = simple_tokenize(text)
        print(f"Tokens: {tokens}")
        
        # Without stop words
        tokens_no_stop = remove_stopwords(tokens)
        print(f"No stopwords: {tokens_no_stop}")
        
        # With stemming
        tokens_stemmed = stem_tokens(tokens_no_stop)
        print(f"Stemmed: {tokens_stemmed}")
        
        # Full pipeline
        processed = tokenize_and_process(text)
        print(f"Full pipeline: {processed}")
