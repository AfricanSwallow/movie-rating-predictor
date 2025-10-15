"""
IMDB Reviews dataset collector from Hugging Face - Direct file download approach.
Downloads individual movie review JSON files and combines them into a single dataset.
"""

import os
import pandas as pd
import json
from huggingface_hub import hf_hub_download, HfApi
from typing import Optional, List
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IMDBReviewsDirectCollector:
    """Direct collector for IMDB Reviews dataset from Hugging Face."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the collector.
        
        Args:
            data_dir: Directory to save the downloaded data
        """
        self.data_dir = data_dir
        self.dataset_name = "Daksh0505/IMDB-Reviews"
        self.output_file = os.path.join(data_dir, "imdb_reviews.csv")
        self.api = HfApi()
        
    def get_review_files(self) -> List[str]:
        """
        Get list of all review JSON files in the dataset.
        
        Returns:
            List of review file names
        """
        try:
            files = self.api.list_repo_files(self.dataset_name, repo_type="dataset")
            review_files = [f for f in files if f.endswith('_reviews.json')]
            logger.info(f"📁 Found {len(review_files)} review files")
            return review_files
        except Exception as e:
            logger.error(f"❌ Error getting file list: {e}")
            raise
    
    def download_and_parse_file(self, filename: str) -> List[dict]:
        """
        Download and parse a single review file.
        
        Args:
            filename: Name of the file to download
            
        Returns:
            List of review dictionaries
        """
        try:
            # Download the file
            downloaded_file = hf_hub_download(
                repo_id=self.dataset_name,
                filename=filename,
                repo_type="dataset"
            )
            
            # Parse the JSON
            with open(downloaded_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            movie_id = data.get('movie_id', '')
            reviews = data.get('reviews', [])
            
            # Flatten the reviews with movie_id
            flattened_reviews = []
            for review in reviews:
                flattened_review = {
                    'movie_id': movie_id,
                    'title': review.get('title', ''),
                    'review': review.get('review', ''),
                    'rating': review.get('rating', '')
                }
                flattened_reviews.append(flattened_review)
            
            return flattened_reviews
            
        except Exception as e:
            logger.warning(f"⚠️ Error processing file {filename}: {e}")
            return []
    
    def download_all_reviews(self, max_files: Optional[int] = None) -> pd.DataFrame:
        """
        Download and combine all review files.
        
        Args:
            max_files: Optional limit on number of files to process (for testing)
            
        Returns:
            pandas DataFrame containing all reviews
        """
        logger.info(f"🤗 Downloading IMDB Reviews dataset from {self.dataset_name}")
        
        # Get list of files
        review_files = self.get_review_files()
        
        # Limit files if specified
        if max_files:
            review_files = review_files[:max_files]
            logger.info(f"📋 Limited to {max_files} files for testing")
        
        # Download and process all files
        all_reviews = []
        
        logger.info(f"📥 Downloading and processing {len(review_files)} files...")
        for filename in tqdm(review_files, desc="Processing files"):
            file_reviews = self.download_and_parse_file(filename)
            all_reviews.extend(file_reviews)
        
        # Convert to DataFrame
        if all_reviews:
            df = pd.DataFrame(all_reviews)
            logger.info(f"✅ Successfully processed {len(df)} reviews from {len(review_files)} movies")
        else:
            df = pd.DataFrame()
            logger.warning("⚠️ No reviews were successfully processed")
        
        return df
    
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
            'total_reviews': len(df),
            'unique_movies': df['movie_id'].nunique() if 'movie_id' in df.columns else 0,
            'columns': list(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
        }
        
        # Rating distribution
        if 'rating' in df.columns:
            rating_counts = df['rating'].value_counts().sort_index()
            info['rating_distribution'] = rating_counts.to_dict()
        
        # Text length analysis
        if 'review' in df.columns:
            review_lengths = df['review'].str.len()
            info['review_length_stats'] = {
                'mean': review_lengths.mean(),
                'median': review_lengths.median(),
                'min': review_lengths.min(),
                'max': review_lengths.max()
            }
        
        # Check for missing data
        missing_data = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_data[col] = missing_count
        info['missing_data'] = missing_data
        
        return info
    
    def download_and_save(self, max_files: Optional[int] = None, 
                         filename: Optional[str] = None) -> tuple[pd.DataFrame, str]:
        """
        Download and save the dataset in one step.
        
        Args:
            max_files: Optional limit on number of files to process
            filename: Optional custom filename
            
        Returns:
            Tuple of (DataFrame, saved_file_path)
        """
        # Download dataset
        df = self.download_all_reviews(max_files=max_files)
        
        if df.empty:
            raise ValueError("No data was successfully downloaded")
        
        # Save dataset
        saved_path = self.save_dataset(df, filename=filename)
        
        # Get and display dataset info
        info = self.get_dataset_info(df)
        logger.info("📊 Dataset Information:")
        for key, value in info.items():
            if isinstance(value, dict) and len(str(value)) > 200:
                logger.info(f"  {key}: {type(value).__name__} with {len(value)} entries")
            else:
                logger.info(f"  {key}: {value}")
        
        return df, saved_path


def main():
    """Main function to download the IMDB Reviews dataset."""
    collector = IMDBReviewsDirectCollector()
    
    try:
        # Download and save the dataset
        # Use max_files=10 for testing, remove for full dataset
        df, saved_path = collector.download_and_save(max_files=None)  # Set to None for full dataset
        
        print(f"\n🎉 Successfully downloaded IMDB Reviews dataset!")
        print(f"📁 Saved to: {saved_path}")
        print(f"📊 Dataset shape: {df.shape}")
        
        # Show basic statistics
        print(f"\n📋 Dataset Overview:")
        print(f"Columns: {list(df.columns)}")
        print("\nSample data:")
        print(df.head())
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())