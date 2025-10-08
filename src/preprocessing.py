"""
Data preprocessing module for movie rating prediction.
Handles data cleaning, normalization, and preprocessing for IMDb datasets.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder

class MovieDataPreprocessor:
    """Preprocess movie data for machine learning."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
    
    def clean_imdb_data(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Clean IMDb datasets.
        
        Args:
            dataframes: Dictionary of raw IMDb DataFrames
            
        Returns:
            Dictionary of cleaned DataFrames
        """
        cleaned = {}
        
        # Clean title basics
        if 'title_basics' in dataframes:
            df = dataframes['title_basics'].copy()
            
            # Replace '\\N' with NaN
            df = df.replace('\\N', np.nan)
            
            # Convert year columns to numeric
            df['startYear'] = pd.to_numeric(df['startYear'], errors='coerce')
            df['endYear'] = pd.to_numeric(df['endYear'], errors='coerce')
            df['runtimeMinutes'] = pd.to_numeric(df['runtimeMinutes'], errors='coerce')
            
            # Filter for movies only
            df = df[df['titleType'] == 'movie']
            
            # Filter reasonable years (1900-2030)
            df = df[(df['startYear'] >= 1900) & (df['startYear'] <= 2030)]
            
            # Filter reasonable runtime (10-600 minutes)
            df = df[(df['runtimeMinutes'] >= 10) & (df['runtimeMinutes'] <= 600)]
            
            cleaned['title_basics'] = df
        
        # Clean title ratings
        if 'title_ratings' in dataframes:
            df = dataframes['title_ratings'].copy()
            
            # Convert to numeric
            df['averageRating'] = pd.to_numeric(df['averageRating'], errors='coerce')
            df['numVotes'] = pd.to_numeric(df['numVotes'], errors='coerce')
            
            # Filter minimum vote threshold
            df = df[df['numVotes'] >= 100]  # At least 100 votes
            
            cleaned['title_ratings'] = df
        
        # Clean name basics
        if 'name_basics' in dataframes:
            df = dataframes['name_basics'].copy()
            
            # Replace '\\N' with NaN
            df = df.replace('\\N', np.nan)
            
            # Convert years to numeric
            df['birthYear'] = pd.to_numeric(df['birthYear'], errors='coerce')
            df['deathYear'] = pd.to_numeric(df['deathYear'], errors='coerce')
            
            cleaned['name_basics'] = df
        
        # Clean crew data
        if 'title_crew' in dataframes:
            df = dataframes['title_crew'].copy()
            df = df.replace('\\N', np.nan)
            cleaned['title_crew'] = df
        
        # Clean principals data
        if 'title_principals' in dataframes:
            df = dataframes['title_principals'].copy()
            df = df.replace('\\N', np.nan)
            df['ordering'] = pd.to_numeric(df['ordering'], errors='coerce')
            cleaned['title_principals'] = df
        
        # Clean akas (regional) data
        if 'title_akas' in dataframes:
            df = dataframes['title_akas'].copy()
            df = df.replace('\\N', np.nan)
            
            # Standardize column name for consistency
            if 'titleId' in df.columns:
                df = df.rename(columns={'titleId': 'tconst'})
            
            # Convert ordering to numeric
            if 'ordering' in df.columns:
                df['ordering'] = pd.to_numeric(df['ordering'], errors='coerce')
            
            # Convert isOriginalTitle to boolean
            if 'isOriginalTitle' in df.columns:
                df['isOriginalTitle'] = df['isOriginalTitle'].map({'1': True, '0': False, 1: True, 0: False})
            
            cleaned['title_akas'] = df
        
        return cleaned
    
    def extract_genres(self, genre_string: str) -> List[str]:
        """
        Extract individual genres from genre string.
        
        Args:
            genre_string: Comma-separated genre string
            
        Returns:
            List of individual genres
        """
        if pd.isna(genre_string):
            return []
        
        genres = [genre.strip() for genre in genre_string.split(',')]
        return [genre for genre in genres if genre and genre != '\\N']
    
    def create_genre_features(self, df: pd.DataFrame, genre_col: str = 'genres') -> pd.DataFrame:
        """
        Create one-hot encoded genre features.
        
        Args:
            df: DataFrame with genre column
            genre_col: Name of the genre column
            
        Returns:
            DataFrame with genre features
        """
        # Extract all unique genres
        all_genres = set()
        for genres_str in df[genre_col].dropna():
            genres = self.extract_genres(genres_str)
            all_genres.update(genres)
        
        # Create binary features for each genre
        for genre in sorted(all_genres):
            df[f'genre_{genre.lower().replace("-", "_")}'] = df[genre_col].apply(
                lambda x: 1 if genre in self.extract_genres(x) else 0
            )
        
        return df
    
    def normalize_ratings(self, ratings: pd.Series, source_scale: Tuple[float, float] = (1, 10),
                         target_scale: Tuple[float, float] = (0, 1)) -> pd.Series:
        """
        Normalize ratings to a common scale.
        
        Args:
            ratings: Series of ratings
            source_scale: Original scale (min, max)
            target_scale: Target scale (min, max)
            
        Returns:
            Normalized ratings
        """
        source_min, source_max = source_scale
        target_min, target_max = target_scale
        
        # Normalize to 0-1 first
        normalized = (ratings - source_min) / (source_max - source_min)
        
        # Scale to target range
        scaled = normalized * (target_max - target_min) + target_min
        
        return scaled.clip(target_min, target_max)
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: Dictionary mapping column names to strategies
                     ('drop', 'mean', 'median', 'mode', 'forward_fill', 'constant')
            
        Returns:
            DataFrame with handled missing values
        """
        if strategy is None:
            strategy = {}
        
        df_processed = df.copy()
        
        for column in df_processed.columns:
            if df_processed[column].isna().any():
                method = strategy.get(column, 'median' if df_processed[column].dtype in ['int64', 'float64'] else 'mode')
                
                if method == 'drop':
                    df_processed = df_processed.dropna(subset=[column])
                elif method == 'mean':
                    df_processed[column].fillna(df_processed[column].mean(), inplace=True)
                elif method == 'median':
                    df_processed[column].fillna(df_processed[column].median(), inplace=True)
                elif method == 'mode':
                    mode_value = df_processed[column].mode()
                    if len(mode_value) > 0:
                        df_processed[column].fillna(mode_value[0], inplace=True)
                elif method == 'forward_fill':
                    df_processed[column].fillna(method='ffill', inplace=True)
                elif method == 'constant':
                    df_processed[column].fillna('Unknown', inplace=True)
        
        return df_processed
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: Input DataFrame
            categorical_columns: List of categorical column names
            
        Returns:
            DataFrame with encoded features
        """
        df_encoded = df.copy()
        
        for column in categorical_columns:
            if column in df_encoded.columns:
                if column not in self.encoders:
                    self.encoders[column] = LabelEncoder()
                    df_encoded[f'{column}_encoded'] = self.encoders[column].fit_transform(
                        df_encoded[column].astype(str)
                    )
                else:
                    df_encoded[f'{column}_encoded'] = self.encoders[column].transform(
                        df_encoded[column].astype(str)
                    )
        
        return df_encoded
    
    def combine_imdb_datasets(self, dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine multiple IMDb datasets into a single feature-rich dataset.
        
        Args:
            dataframes: Dictionary of cleaned IMDb DataFrames
            
        Returns:
            Combined DataFrame ready for machine learning
        """
        # Start with title basics as the main dataset
        main_df = dataframes['title_basics'].copy()
        
        # Add ratings data
        if 'title_ratings' in dataframes:
            ratings_df = dataframes['title_ratings'][['tconst', 'averageRating', 'numVotes']]
            main_df = main_df.merge(ratings_df, on='tconst', how='left')
        
        # Add crew information (directors and writers)
        if 'title_crew' in dataframes:
            crew_df = dataframes['title_crew'].copy()
            # Count number of directors and writers
            crew_df['num_directors'] = crew_df['directors'].apply(
                lambda x: len(x.split(',')) if pd.notna(x) else 0
            )
            crew_df['num_writers'] = crew_df['writers'].apply(
                lambda x: len(x.split(',')) if pd.notna(x) else 0
            )
            main_df = main_df.merge(
                crew_df[['tconst', 'directors', 'writers', 'num_directors', 'num_writers']], 
                on='tconst', how='left'
            )
        
        # Add cast information
        if 'title_principals' in dataframes:
            principals_df = dataframes['title_principals'].copy()
            
            # Count number of actors/actresses and other roles
            cast_stats = principals_df.groupby('tconst').agg({
                'nconst': 'count',  # total cast/crew count
                'category': lambda x: (x.isin(['actor', 'actress'])).sum()  # actor count
            }).rename(columns={'nconst': 'total_cast_crew', 'category': 'num_actors'})
            
            main_df = main_df.merge(cast_stats, on='tconst', how='left')
        
        # Add regional/language data
        if 'title_akas' in dataframes:
            akas_df = dataframes['title_akas'].copy()
            
            # Aggregate regional data for each movie
            regional_stats = akas_df.groupby('tconst').agg({
                'region': lambda x: '|'.join(x.dropna().unique()),  # All regions
                'language': lambda x: '|'.join(x.dropna().unique()),  # All languages
                'title': 'count'  # Number of regional variants
            }).rename(columns={'title': 'num_regional_variants'})
            
            # Count unique regions and languages
            regional_stats['num_regions'] = regional_stats['region'].apply(
                lambda x: len(set(x.split('|'))) if pd.notna(x) and x else 0
            )
            regional_stats['num_languages'] = regional_stats['language'].apply(
                lambda x: len(set(x.split('|'))) if pd.notna(x) and x else 0
            )
            
            main_df = main_df.merge(regional_stats, on='tconst', how='left')
            
            # Fill NaN for movies without regional data
            main_df['num_regional_variants'] = main_df['num_regional_variants'].fillna(0)
            main_df['num_regions'] = main_df['num_regions'].fillna(0)
            main_df['num_languages'] = main_df['num_languages'].fillna(0)
        
        # Create additional features
        main_df = self.create_genre_features(main_df, 'genres')
        
        # Calculate movie age
        current_year = pd.Timestamp.now().year
        main_df['movie_age'] = current_year - main_df['startYear']
        
        return main_df
    
    def extract_top_people(self, dataframes: Dict[str, pd.DataFrame], 
                          role_type: str = 'director', top_n: int = 100) -> List[str]:
        """
        Extract top people (directors, actors, etc.) based on number of movies.
        
        Args:
            dataframes: Dictionary of IMDb DataFrames
            role_type: 'director', 'actor', 'actress', etc.
            top_n: Number of top people to return
            
        Returns:
            List of top person IDs
        """
        if role_type == 'director' and 'title_crew' in dataframes:
            # Extract directors from crew data
            crew_df = dataframes['title_crew'].dropna(subset=['directors'])
            all_directors = []
            for directors_str in crew_df['directors']:
                directors = [d.strip() for d in directors_str.split(',')]
                all_directors.extend(directors)
            
            director_counts = pd.Series(all_directors).value_counts()
            return director_counts.head(top_n).index.tolist()
        
        elif role_type in ['actor', 'actress'] and 'title_principals' in dataframes:
            # Extract actors from principals data
            principals_df = dataframes['title_principals']
            actors_df = principals_df[principals_df['category'].isin(['actor', 'actress'])]
            actor_counts = actors_df['nconst'].value_counts()
            return actor_counts.head(top_n).index.tolist()
        
        return []
    
    def create_people_features(self, df: pd.DataFrame, dataframes: Dict[str, pd.DataFrame],
                              top_directors: int = 20, top_actors: int = 30) -> pd.DataFrame:
        """
        Create features based on top directors and actors.
        
        Args:
            df: Main movie DataFrame
            dataframes: Dictionary of IMDb DataFrames
            top_directors: Number of top directors to create features for
            top_actors: Number of top actors to create features for
            
        Returns:
            DataFrame with people-based features
        """
        print(f"  Creating features for top {top_directors} directors and {top_actors} actors...")
        df_with_people = df.copy()
        
        # Get top directors and actors
        top_director_ids = self.extract_top_people(dataframes, 'director', top_directors)
        top_actor_ids = self.extract_top_people(dataframes, 'actor', top_actors)
        
        # Create director features using vectorized operations
        print(f"  Processing {len(top_director_ids)} top directors...")
        director_features = {}
        for director_id in top_director_ids:
            director_features[f'director_{director_id}'] = df_with_people['directors'].apply(
                lambda x: 1 if pd.notna(x) and director_id in x else 0
            )
        
        # Create actor features more efficiently
        print(f"  Processing {len(top_actor_ids)} top actors...")
        if 'title_principals' in dataframes:
            principals_df = dataframes['title_principals']
            actors_df = principals_df[
                (principals_df['category'].isin(['actor', 'actress'])) & 
                (principals_df['nconst'].isin(top_actor_ids))
            ]
            
            # Create a mapping of movies to actors for faster lookup
            movie_to_actors = actors_df.groupby('tconst')['nconst'].apply(set).to_dict()
            
            # Create actor features more efficiently
            actor_features = {}
            for actor_id in top_actor_ids:
                actor_features[f'actor_{actor_id}'] = df_with_people['tconst'].apply(
                    lambda x: 1 if x in movie_to_actors and actor_id in movie_to_actors[x] else 0
                )
        
        # Combine all features at once to avoid fragmentation
        print("  Combining all people features...")
        all_features = {**director_features, **actor_features}
        people_df = pd.DataFrame(all_features, index=df_with_people.index)
        
        # Concatenate with original dataframe
        final_df = pd.concat([df_with_people, people_df], axis=1)
        
        return final_df

def main():
    """Main preprocessing pipeline for IMDb data."""
    preprocessor = MovieDataPreprocessor()
    
    print("🔧 Starting IMDb data preprocessing...")
    
    # Check if IMDb data exists
    data_dir = 'data/processed'
    required_files = [
        'imdb_title_basics.parquet',
        'imdb_title_ratings.parquet', 
        'imdb_title_crew.parquet',
        'imdb_name_basics.parquet',
        'imdb_title_principals.parquet'
    ]
    
    # Optional files that don't block the pipeline
    optional_files = [
        'imdb_title_akas.parquet'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing IMDb data files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n📥 Please run data collection first:")
        print("   python src/data_collection.py")
        return
    
    # Load IMDb datasets
    print("📊 Loading IMDb datasets...")
    dataframes = {}
    
    # Load required files
    for file in required_files:
        name = file.replace('imdb_', '').replace('.parquet', '')
        try:
            df = pd.read_parquet(os.path.join(data_dir, file))
            dataframes[name] = df
            print(f"   ✅ Loaded {name}: {len(df):,} records")
        except Exception as e:
            print(f"   ❌ Error loading {file}: {e}")
    
    # Load optional files
    for file in optional_files:
        name = file.replace('imdb_', '').replace('.parquet', '')
        filepath = os.path.join(data_dir, file)
        if os.path.exists(filepath):
            try:
                df = pd.read_parquet(filepath)
                dataframes[name] = df
                print(f"   ✅ Loaded {name} (optional): {len(df):,} records")
            except Exception as e:
                print(f"   ⚠️  Error loading optional {file}: {e}")
        else:
            print(f"   ℹ️  Optional file {file} not found, skipping...")
    
    if not dataframes:
        print("❌ No IMDb data could be loaded")
        return
    
    # Clean data
    print("\n🧹 Cleaning IMDb data...")
    cleaned_data = preprocessor.clean_imdb_data(dataframes)
    
    # Save cleaned individual datasets (for EDA)
    print("💾 Saving cleaned individual datasets...")
    for name, df in cleaned_data.items():
        output_file = f'data/processed/imdb_{name}.parquet'
        df.to_parquet(output_file)
        print(f"   ✅ Saved {output_file}: {len(df):,} records")
    
    # Combine datasets
    print("🔗 Combining IMDb datasets...")
    combined_df = preprocessor.combine_imdb_datasets(cleaned_data)
    
    # Save intermediate result
    combined_df.to_parquet('data/processed/imdb_combined_basic.parquet')
    print(f"  💾 Saved basic combined dataset: {len(combined_df):,} movies with {len(combined_df.columns)} features")
    
    # Create people-based features (reduced number for performance)
    print("👥 Creating people-based features...")
    final_df = preprocessor.create_people_features(combined_df, cleaned_data, 
                                                 top_directors=10, top_actors=20)
    
    # Handle missing values
    print("🔧 Handling missing values...")
    final_df = preprocessor.handle_missing_values(final_df)
    
    # Save processed data
    output_path = 'data/processed_movie_features.csv'
    final_df.to_csv(output_path, index=False)
    print(f"\n💾 Saved processed features to {output_path}")
    
    # Save feature information
    feature_info = {
        'total_features': len(final_df.columns),
        'total_movies': len(final_df),
        'numeric_features': final_df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_features': final_df.select_dtypes(include=['object']).columns.tolist(),
        'missing_values': final_df.isnull().sum().to_dict()
    }
    
    import json
    with open('data/feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2, default=str)
    
    print(f"📋 Feature summary:")
    print(f"   - Total movies: {len(final_df):,}")
    print(f"   - Total features: {len(final_df.columns)}")
    print(f"   - Numeric features: {len(feature_info['numeric_features'])}")
    print(f"   - Categorical features: {len(feature_info['categorical_features'])}")
    
    print("\n🎉 IMDb data preprocessing completed!")

if __name__ == "__main__":
    main()