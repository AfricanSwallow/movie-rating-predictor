"""
IMDB Reviews dataset collector from Hugging Face.
Downloads and processes the Daksh0505/IMDB-Reviews dataset for text analysis.
"""

import os
import pandas as pd
from datasets import load_dataset
from typing import Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IMDBReviewsCollector:
    """Collector for IMDB Reviews dataset from Hugging Face."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the collector.
        
        Args:
            data_dir: Directory to save the downloaded data
        """
        self.data_dir = data_dir
        self.dataset_name = "Daksh0505/IMDB-Reviews"
        self.output_file = os.path.join(data_dir, "imdb_reviews.csv")
        
    def download_dataset(self, cache_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Download the IMDB Reviews dataset from Hugging Face.
        
        Args:
            cache_dir: Optional cache directory for Hugging Face datasets
            
        Returns:
            pandas DataFrame containing the reviews data
        """
        logger.info(f"🤗 Downloading IMDB Reviews dataset from {self.dataset_name}")
        
        try:
            # Load dataset from Hugging Face
            dataset = load_dataset(
                self.dataset_name,
                cache_dir=cache_dir
            )
            
            logger.info(f"✅ Dataset loaded successfully!")
            logger.info(f"📊 Dataset structure: {dataset}")
            
            # Check what splits are available
            available_splits = list(dataset.keys())
            logger.info(f"📋 Available splits: {available_splits}")
            
            # Convert to pandas DataFrame
            # The dataset contains movie_id and reviews columns
            # We need to process the reviews JSON data
            all_data = []
            
            for split in available_splits:
                split_df = dataset[split].to_pandas()
                logger.info(f"📊 {split} split: {len(split_df)} rows")
                
                # Process the reviews column which contains JSON data
                expanded_reviews = []
                for idx, row in split_df.iterrows():
                    movie_id = row['movie_id']
                    reviews_data = row['reviews']
                    
                    # Parse reviews if it's a string
                    if isinstance(reviews_data, str):
                        import json
                        try:
                            reviews_data = json.loads(reviews_data)
                        except json.JSONDecodeError:
                            logger.warning(f"Could not parse reviews for movie {movie_id}")
                            continue
                    
                    # Extract individual reviews
                    if isinstance(reviews_data, list):
                        for review in reviews_data:
                            expanded_reviews.append({
                                'movie_id': movie_id,
                                'title': review.get('title', ''),
                                'review': review.get('review', ''),
                                'rating': review.get('rating', ''),
                                'split': split
                            })
                    elif isinstance(reviews_data, dict):
                        # Single review case
                        expanded_reviews.append({
                            'movie_id': movie_id,
                            'title': reviews_data.get('title', ''),
                            'review': reviews_data.get('review', ''),
                            'rating': reviews_data.get('rating', ''),
                            'split': split
                        })
                
                # Convert expanded reviews to DataFrame
                if expanded_reviews:
                    expanded_df = pd.DataFrame(expanded_reviews)
                    all_data.append(expanded_df)
                    logger.info(f"📊 {split} split expanded: {len(expanded_df)} individual reviews")
            
            # Combine all splits
            if len(all_data) > 1:
                combined_df = pd.concat(all_data, ignore_index=True)
                logger.info(f"📊 Combined dataset: {len(combined_df)} total reviews")
            elif len(all_data) == 1:
                combined_df = all_data[0]
            else:
                raise ValueError("No data was successfully processed")
            
            # Display dataset info
            logger.info(f"📋 Dataset columns: {list(combined_df.columns)}")
            logger.info(f"📊 Dataset shape: {combined_df.shape}")
            
            # Show sample data
            logger.info("🔍 Sample data:")
            print(combined_df.head())
            
            return combined_df
            
        except Exception as e:
            logger.error(f"❌ Error downloading dataset: {e}")
            raise
    
    def save_dataset(self, df: pd.DataFrame, filename: Optional[str] = None) -> str:
        """
        Save the dataset to CSV file.
        
        Args:
            df: DataFrame to save
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        # Create directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Use custom filename if provided
        if filename:
            output_path = os.path.join(self.data_dir, filename)
        else:
            output_path = self.output_file
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"💾 Dataset saved to: {output_path}")
        
        return output_path
    
    def get_dataset_info(self, df: pd.DataFrame) -> dict:
        """
        Get information about the dataset for analysis.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            'total_rows': len(df),
            'columns': list(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'dtypes': df.dtypes.to_dict()
        }
        
        # Check for text columns (likely review text)
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if this looks like text data
                sample_values = df[col].dropna().head()
                if len(sample_values) > 0:
                    avg_length = sample_values.astype(str).str.len().mean()
                    if avg_length > 50:  # Likely text if average length > 50 chars
                        text_columns.append(col)
        
        info['text_columns'] = text_columns
        
        # Check for rating/sentiment columns
        rating_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['rating', 'score', 'sentiment', 'label']):
                rating_columns.append(col)
        
        info['rating_columns'] = rating_columns
        
        return info
    
    def download_and_save(self, cache_dir: Optional[str] = None, 
                         filename: Optional[str] = None) -> tuple[pd.DataFrame, str]:
        """
        Download and save the dataset in one step.
        
        Args:
            cache_dir: Optional cache directory for Hugging Face datasets
            filename: Optional custom filename
            
        Returns:
            Tuple of (DataFrame, saved_file_path)
        """
        # Download dataset
        df = self.download_dataset(cache_dir=cache_dir)
        
        # Save dataset
        saved_path = self.save_dataset(df, filename=filename)
        
        # Get and display dataset info
        info = self.get_dataset_info(df)
        logger.info("📊 Dataset Information:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")
        
        return df, saved_path


def main():
    """Main function to download the IMDB Reviews dataset."""
    collector = IMDBReviewsCollector()
    
    try:
        # Download and save the dataset
        df, saved_path = collector.download_and_save()
        
        print(f"\n🎉 Successfully downloaded IMDB Reviews dataset!")
        print(f"📁 Saved to: {saved_path}")
        print(f"📊 Dataset shape: {df.shape}")
        
        # Show basic statistics
        print(f"\n📋 Dataset Overview:")
        print(df.info())
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())