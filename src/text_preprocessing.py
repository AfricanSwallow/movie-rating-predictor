"""
Text Preprocessing Module for Movie Review Analysis

This module provides functions for cleaning and preprocessing movie reviews:
- HTML tag removal
- URL removal
- Special character handling
- Text normalization
- Tokenization
- Stop word removal
- Stemming/Lemmatization
"""

import re
import string
from typing import List, Optional
import pandas as pd
from bs4 import BeautifulSoup


def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags from text using BeautifulSoup.
    
    Args:
        text: Input text that may contain HTML tags
        
    Returns:
        Text with HTML tags removed
    """
    if pd.isna(text):
        return ""
    
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_urls(text: str) -> str:
    """
    Remove URLs from text.
    
    Args:
        text: Input text that may contain URLs
        
    Returns:
        Text with URLs removed
    """
    if pd.isna(text):
        return ""
    
    # Remove http/https URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove www URLs
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    return text


def remove_special_characters(text: str, keep_punctuation: bool = False) -> str:
    """
    Remove special characters from text.
    
    Args:
        text: Input text
        keep_punctuation: If True, keeps basic punctuation marks (.,!?)
        
    Returns:
        Text with special characters removed
    """
    if pd.isna(text):
        return ""
    
    if keep_punctuation:
        # Keep letters, numbers, spaces, and basic punctuation
        pattern = r'[^a-zA-Z0-9\s.,!?\'\"-]'
    else:
        # Keep only letters, numbers, and spaces
        pattern = r'[^a-zA-Z0-9\s]'
    
    text = re.sub(pattern, ' ', text)
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace by replacing multiple spaces with single space.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    if pd.isna(text):
        return ""
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading and trailing whitespace
    text = text.strip()
    
    return text


def convert_to_lowercase(text: str) -> str:
    """
    Convert text to lowercase.
    
    Args:
        text: Input text
        
    Returns:
        Lowercase text
    """
    if pd.isna(text):
        return ""
    
    return text.lower()


def expand_contractions(text: str) -> str:
    """
    Expand common English contractions (e.g., don't -> do not).
    
    Args:
        text: Input text
        
    Returns:
        Text with expanded contractions
    """
    if pd.isna(text):
        return ""
    
    # Common contractions dictionary
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "could've": "could have",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "how'd": "how did",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'll": "i will",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'll": "it will",
        "it's": "it is",
        "let's": "let us",
        "might've": "might have",
        "must've": "must have",
        "needn't": "need not",
        "shan't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "that'd": "that would",
        "that's": "that is",
        "there'd": "there would",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'll": "we will",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "where'd": "where did",
        "where's": "where is",
        "who'll": "who will",
        "who's": "who is",
        "won't": "will not",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have"
    }
    
    # Convert to lowercase for matching
    text_lower = text.lower()
    
    # Replace contractions
    for contraction, expansion in contractions.items():
        text_lower = text_lower.replace(contraction, expansion)
    
    return text_lower


def clean_text(text: str, 
               remove_html: bool = True,
               remove_urls_flag: bool = True,
               expand_contractions_flag: bool = True,
               to_lowercase: bool = True,
               remove_special_chars: bool = True,
               keep_punctuation: bool = False) -> str:
    """
    Apply all text cleaning steps in sequence.
    
    Args:
        text: Input text to clean
        remove_html: Remove HTML tags
        remove_urls_flag: Remove URLs
        expand_contractions_flag: Expand contractions
        to_lowercase: Convert to lowercase
        remove_special_chars: Remove special characters
        keep_punctuation: Keep basic punctuation marks
        
    Returns:
        Cleaned text
    """
    if pd.isna(text):
        return ""
    
    # Apply cleaning steps
    if remove_html:
        text = remove_html_tags(text)
    
    if remove_urls_flag:
        text = remove_urls(text)
    
    if expand_contractions_flag:
        text = expand_contractions(text)
    
    if to_lowercase:
        text = convert_to_lowercase(text)
    
    if remove_special_chars:
        text = remove_special_characters(text, keep_punctuation=keep_punctuation)
    
    # Always normalize whitespace at the end
    text = normalize_whitespace(text)
    
    return text


def clean_dataframe_column(df: pd.DataFrame, 
                           column_name: str,
                           output_column: Optional[str] = None,
                           **kwargs) -> pd.DataFrame:
    """
    Apply text cleaning to a DataFrame column.
    
    Args:
        df: Input DataFrame
        column_name: Name of column to clean
        output_column: Name for output column (default: column_name + '_cleaned')
        **kwargs: Additional arguments to pass to clean_text()
        
    Returns:
        DataFrame with cleaned text column added
    """
    if output_column is None:
        output_column = f"{column_name}_cleaned"
    
    print(f"Cleaning text in column '{column_name}'...")
    df[output_column] = df[column_name].apply(lambda x: clean_text(x, **kwargs))
    print(f"✓ Cleaned text saved to column '{output_column}'")
    
    return df


if __name__ == "__main__":
    # Test the cleaning functions
    sample_texts = [
        "<p>This is a <b>great</b> movie!</p>",
        "Check out this link: http://example.com for more info.",
        "I can't believe how amazing this film was!!! It's the best movie I've ever seen.",
        "The special effects were @#$% incredible & the story was compelling.",
    ]
    
    print("Text Cleaning Examples:")
    print("=" * 80)
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\nOriginal {i}: {text}")
        cleaned = clean_text(text)
        print(f"Cleaned {i}:  {cleaned}")
