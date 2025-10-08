"""
Quick data exploration script to see what data we've collected.
"""

import pandas as pd
import os

def explore_collected_data():
    """Explore the data we've collected."""
    data_dir = 'data/processed'
    
    print("🎬 MOVIE DATA COLLECTION SUMMARY")
    print("=" * 50)
    
    # Check MovieLens data
    print("\n📊 MOVIELENS DATASET:")
    
    # Movies
    movies_df = pd.read_parquet(f'{data_dir}/movielens_movies.parquet')
    print(f"   🎭 Movies: {len(movies_df):,} records")
    print(f"       Sample: {movies_df['title'].iloc[0]}")
    print(f"       Genres: {movies_df['genres'].iloc[0]}")
    
    # Ratings
    ratings_df = pd.read_parquet(f'{data_dir}/movielens_ratings.parquet')
    print(f"   ⭐ Ratings: {len(ratings_df):,} records")
    print(f"       Rating range: {ratings_df['rating'].min():.1f} - {ratings_df['rating'].max():.1f}")
    print(f"       Average rating: {ratings_df['rating'].mean():.2f}")
    
    # Check TMDb data
    if os.path.exists(f'{data_dir}/tmdb_popular_movies.parquet'):
        print("\n📊 TMDB DATASET:")
        tmdb_df = pd.read_parquet(f'{data_dir}/tmdb_popular_movies.parquet')
        print(f"   🎬 Popular Movies: {len(tmdb_df):,} records")
        if 'title' in tmdb_df.columns:
            print(f"       Sample: {tmdb_df['title'].iloc[0]}")
        if 'vote_average' in tmdb_df.columns:
            print(f"       TMDb rating range: {tmdb_df['vote_average'].min():.1f} - {tmdb_df['vote_average'].max():.1f}")
    
    print("\n✅ DATA READY FOR ANALYSIS!")
    print("\n📋 Next steps:")
    print("   1. Open notebooks/02_exploratory_data_analysis.ipynb")
    print("   2. Replace sample data with this real data")
    print("   3. Analyze rating patterns by genre, year, etc.")
    print("   4. Build your prediction model!")
    
    return movies_df, ratings_df

if __name__ == "__main__":
    movies, ratings = explore_collected_data()