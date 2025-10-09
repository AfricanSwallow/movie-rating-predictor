"""
Data collection module for movie rating prediction.
Handles data collection from IMDb datasets.
"""

import pandas as pd
import requests
import os
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

class MovieDataCollector:
    """Collect movie data from IMDb datasets."""
    
    def __init__(self):
        self.base_urls = {
            'imdb_datasets': 'https://datasets.imdbws.com'
        }
    
    def _download_file_if_needed(self, filename: str, data_dir: str) -> bool:
        """
        Download a file from IMDb if it doesn't exist locally.
        
        Args:
            filename: Name of the file to download
            data_dir: Directory to save the file
            
        Returns:
            True if file exists or was downloaded successfully, False otherwise
        """
        filepath = os.path.join(data_dir, filename)
        
        if os.path.exists(filepath):
            print(f"📂 {filename} already exists, skipping download")
            return True
        
        print(f"📥 Downloading {filename} from IMDb (free public dataset)...")
        url = f"{self.base_urls['imdb_datasets']}/{filename}"
        
        try:
            # Download file with progress
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            os.makedirs(data_dir, exist_ok=True)
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\r  Progress: {progress:.1f}%", end='', flush=True)
            
            print(f"\n  ✅ Downloaded {filename}")
            return True
            
        except requests.RequestException as e:
            print(f"  ❌ Error downloading {filename}: {e}")
            return False
    
    def collect_imdb_datasets(self, data_dir: str = 'data/raw', movies_only: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Download and load IMDb datasets (No API key required - public datasets).
        
        These are updated daily and contain:
        - title.basics.tsv.gz: Basic movie info (title, year, genres, runtime)
        - title.ratings.tsv.gz: IMDb ratings and vote counts
        - title.crew.tsv.gz: Directors and writers
        - name.basics.tsv.gz: People info (actors, directors)
        - title.principals.tsv.gz: Cast and crew roles (filtered for movies only)
        - title.akas.tsv.gz: Regional titles, languages, and countries
        
        Args:
            data_dir: Directory containing the raw datasets
            movies_only: If True, filter to movies only to reduce memory usage
        
        Returns:
            Dictionary of DataFrames with IMDb data
        """
        datasets = {
            'title_basics': 'title.basics.tsv.gz',
            'title_ratings': 'title.ratings.tsv.gz',
            'title_crew': 'title.crew.tsv.gz',
            'name_basics': 'name.basics.tsv.gz',
            'title_principals': 'title.principals.tsv.gz',
            'title_akas': 'title.akas.tsv.gz'
        }
        
        dataframes = {}
        
        # First, load title_basics to get movie IDs if filtering for movies only
        movie_ids = None
        if movies_only:
            print("📊 Loading title_basics to filter for movies...")
            filename = datasets['title_basics']
            filepath = os.path.join(data_dir, filename)
            
            # Download if needed
            if not self._download_file_if_needed(filename, data_dir):
                print("  ❌ Failed to download title_basics - cannot proceed")
                return {}
            
            try:
                df = pd.read_csv(filepath, sep='\t', low_memory=False, na_values=['\\N'])
                # Filter for movies only
                movies_df = df[df['titleType'] == 'movie'].copy()
                movie_ids = set(movies_df['tconst'].tolist())
                dataframes['title_basics'] = movies_df
                print(f"  ✅ Loaded {len(movies_df):,} movies from {len(df):,} total titles")
            except Exception as e:
                print(f"  ❌ Error loading title_basics: {e}")
                return {}
        
        # Load other datasets
        for name, filename in datasets.items():
            if name == 'title_basics' and movies_only:
                continue  # Already loaded
            
            filepath = os.path.join(data_dir, filename)
            
            # Download if needed
            if not self._download_file_if_needed(filename, data_dir):
                print(f"  ⚠️  Skipping {name} due to download failure")
                continue
            
            print(f"📊 Loading {name}...")
            try:
                if name in ['title_principals', 'title_akas'] and movies_only and movie_ids:
                    # Load large datasets in chunks to filter for movies only
                    print(f"  📊 Processing {filename} in chunks (filtering for movies)...")
                    chunk_list = []
                    chunk_size = 100000
                    
                    for chunk in pd.read_csv(filepath, sep='\t', low_memory=False, 
                                           na_values=['\\N'], chunksize=chunk_size):
                        # Filter for movie IDs only
                        movie_chunk = chunk[chunk['titleId'].isin(movie_ids) if name == 'title_akas' 
                                          else chunk['tconst'].isin(movie_ids)]
                        if not movie_chunk.empty:
                            chunk_list.append(movie_chunk)
                    
                    if chunk_list:
                        df = pd.concat(chunk_list, ignore_index=True)
                        print(f"  ✅ Loaded {len(df):,} movie-related records")
                    else:
                        print(f"  ⚠️  No movie-related records found")
                        continue
                else:
                    df = pd.read_csv(filepath, sep='\t', low_memory=False, na_values=['\\N'])
                    
                    # Filter for movies if movie_ids is available and relevant
                    if movies_only and movie_ids and 'tconst' in df.columns and name != 'name_basics':
                        original_len = len(df)
                        df = df[df['tconst'].isin(movie_ids)]
                        print(f"  ✅ Loaded {len(df):,} movie-related records (filtered from {original_len:,})")
                    else:
                        print(f"  ✅ Loaded {len(df):,} records")
                
                dataframes[name] = df
                
            except Exception as e:
                print(f"  ❌ Error loading {name}: {e}")
                continue
            
        return dataframes

def main():
    """Main data collection pipeline for IMDb datasets."""
    collector = MovieDataCollector()
    
    print("🎬 IMDB MOVIE DATA COLLECTION")
    print("=" * 40)
    print("Collecting comprehensive movie data from IMDb public datasets...")
    print("This includes titles, ratings, cast, crew, and more.")
    print()
    
    try:
        # Collect IMDb datasets
        print("📊 Collecting IMDb datasets...")
        imdb_data = collector.collect_imdb_datasets()
        
        if imdb_data:
            # Save processed data
            os.makedirs('data/processed', exist_ok=True)
            
            for name, df in imdb_data.items():
                if not df.empty:
                    output_path = f'data/processed/imdb_{name}.parquet'
                    df.to_parquet(output_path)
                    print(f"💾 Saved {name} to {output_path}")
        
        print("\n🎉 IMDb data collection completed!")
        
    except Exception as e:
        print(f"❌ Error during data collection: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check your internet connection")
        print("   2. Make sure you have enough disk space")
        print("   3. IMDb datasets are large - ensure stable connection")

if __name__ == "__main__":
    main()