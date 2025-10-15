"""
TMDb Data Integration Module

This module fetches financial data (budget, revenue) and other metadata from TMDb API
and joins it with the existing IMDb dataset.

TMDb provides:
- Budget and revenue (box office) data
- Production companies and countries
- Popularity scores
- TMDb ratings (additional rating source)
- IMDb ID linking for easy joining
"""

import pandas as pd
import requests
import os
import time
import sys
from typing import Dict, Optional, List
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


class TMDbIntegrator:
    """Integrate TMDb financial and metadata with IMDb data."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize TMDb integrator.
        
        Args:
            api_key: TMDb API key. If None, reads from TMDB_API_KEY env variable.
        """
        self.api_key = api_key or os.getenv('TMDB_API_KEY')
        if not self.api_key:
            raise ValueError(
                "TMDb API key required! Get one at https://www.themoviedb.org/settings/api\n"
                "Set it as TMDB_API_KEY environment variable or pass to constructor."
            )
        
        self.base_url = 'https://api.themoviedb.org/3'
        self.session = requests.Session()
        self.session.params = {'api_key': self.api_key}
        
    def find_movie_by_imdb_id(self, imdb_id: str) -> Optional[Dict]:
        """
        Find TMDb movie data using IMDb ID.
        
        Args:
            imdb_id: IMDb ID (e.g., 'tt0111161')
            
        Returns:
            Dictionary with TMDb movie data or None if not found
        """
        try:
            # Use find endpoint to get TMDb ID from IMDb ID
            url = f"{self.base_url}/find/{imdb_id}"
            params = {'external_source': 'imdb_id'}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            movie_results = data.get('movie_results', [])
            
            if not movie_results:
                return None
                
            # Get the first result (usually the correct one)
            movie_summary = movie_results[0]
            tmdb_id = movie_summary['id']
            
            # Fetch full movie details including budget/revenue
            return self.get_movie_details(tmdb_id)
            
        except Exception as e:
            print(f"Error fetching TMDb data for {imdb_id}: {e}")
            return None
    
    def get_movie_details(self, tmdb_id: int) -> Optional[Dict]:
        """
        Get detailed movie information from TMDb.
        
        Args:
            tmdb_id: TMDb movie ID
            
        Returns:
            Dictionary with detailed movie data
        """
        try:
            url = f"{self.base_url}/movie/{tmdb_id}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            print(f"Error fetching TMDb details for ID {tmdb_id}: {e}")
            return None
    
    def extract_tmdb_features(self, tmdb_data: Optional[Dict]) -> Dict:
        """
        Extract relevant features from TMDb data.
        
        Args:
            tmdb_data: Raw TMDb movie data
            
        Returns:
            Dictionary with extracted features
        """
        if not tmdb_data:
            return {
                'tmdb_id': None,
                'budget': None,
                'revenue': None,
                'tmdb_popularity': None,
                'tmdb_vote_average': None,
                'tmdb_vote_count': None,
                'production_countries': None,
                'production_companies': None,
                'spoken_languages': None,
                'original_language': None,
                'tagline': None
            }
        
        # Extract production company names
        companies = tmdb_data.get('production_companies', [])
        company_names = [c['name'] for c in companies] if companies else []
        
        # Extract production country codes
        countries = tmdb_data.get('production_countries', [])
        country_codes = [c['iso_3166_1'] for c in countries] if countries else []
        
        # Extract spoken languages
        languages = tmdb_data.get('spoken_languages', [])
        language_codes = [lang['iso_639_1'] for lang in languages] if languages else []
        
        return {
            'tmdb_id': tmdb_data.get('id'),
            'budget': tmdb_data.get('budget', 0) or 0,  # Convert None to 0
            'revenue': tmdb_data.get('revenue', 0) or 0,
            'tmdb_popularity': tmdb_data.get('popularity'),
            'tmdb_vote_average': tmdb_data.get('vote_average'),
            'tmdb_vote_count': tmdb_data.get('vote_count'),
            'production_countries': ','.join(country_codes) if country_codes else None,
            'production_companies': ','.join(company_names) if company_names else None,
            'spoken_languages': ','.join(language_codes) if language_codes else None,
            'original_language': tmdb_data.get('original_language'),
            'tagline': tmdb_data.get('tagline')
        }
    
    def enrich_imdb_data(self, 
                         imdb_df: pd.DataFrame, 
                         imdb_id_column: str = 'tconst',
                         max_requests: Optional[int] = None,
                         rate_limit_delay: float = 0.25) -> pd.DataFrame:
        """
        Enrich IMDb data with TMDb financial and metadata.
        
        Args:
            imdb_df: DataFrame with IMDb data
            imdb_id_column: Column name containing IMDb IDs
            max_requests: Maximum number of API requests (None = all)
            rate_limit_delay: Delay between requests in seconds (TMDb allows 50/sec)
            
        Returns:
            DataFrame with TMDb data joined
        """
        print(f"🎬 Enriching {len(imdb_df)} movies with TMDb data...")
        
        # Limit number of requests if specified
        df_to_process = imdb_df.head(max_requests) if max_requests else imdb_df
        
        tmdb_data_list = []
        
        for idx, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), 
                            desc="Fetching TMDb data"):
            imdb_id = row[imdb_id_column]
            
            # Fetch TMDb data
            tmdb_data = self.find_movie_by_imdb_id(imdb_id)
            
            # Extract features
            features = self.extract_tmdb_features(tmdb_data)
            features[imdb_id_column] = imdb_id
            
            tmdb_data_list.append(features)
            
            # Rate limiting
            time.sleep(rate_limit_delay)
        
        # Create TMDb DataFrame
        tmdb_df = pd.DataFrame(tmdb_data_list)
        
        # Join with original IMDb data
        enriched_df = imdb_df.merge(tmdb_df, on=imdb_id_column, how='left')
        
        # Calculate ROI (Return on Investment) if we have budget and revenue
        enriched_df['roi'] = enriched_df.apply(
            lambda row: ((row['revenue'] - row['budget']) / row['budget'] * 100) 
            if pd.notna(row['budget']) and row['budget'] > 0 else None,
            axis=1
        )
        
        # Calculate profit
        enriched_df['profit'] = enriched_df['revenue'] - enriched_df['budget']
        
        # Add binary features for financial data availability
        enriched_df['has_budget_data'] = enriched_df['budget'] > 0
        enriched_df['has_revenue_data'] = enriched_df['revenue'] > 0
        
        return enriched_df
    
    def get_dataset_stats(self, df: pd.DataFrame) -> Dict:
        """Get statistics about TMDb data coverage."""
        stats = {
            'total_movies': len(df),
            'with_tmdb_match': df['tmdb_id'].notna().sum(),
            'with_budget': (df['budget'] > 0).sum(),
            'with_revenue': (df['revenue'] > 0).sum(),
            'with_both_financial': ((df['budget'] > 0) & (df['revenue'] > 0)).sum(),
            'avg_budget': df[df['budget'] > 0]['budget'].mean() if (df['budget'] > 0).any() else 0,
            'avg_revenue': df[df['revenue'] > 0]['revenue'].mean() if (df['revenue'] > 0).any() else 0,
            'total_budget': df['budget'].sum(),
            'total_revenue': df['revenue'].sum()
        }
        return stats


def main():
    """Main integration pipeline."""
    print("🎬 TMDb DATA INTEGRATION")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv('TMDB_API_KEY')
    if not api_key:
        print("❌ TMDb API key not found!")
        print("\n📝 To get a TMDb API key:")
        print("   1. Create account at https://www.themoviedb.org/signup")
        print("   2. Go to https://www.themoviedb.org/settings/api")
        print("   3. Request an API key (free for non-commercial use)")
        print("   4. Add to .env file: TMDB_API_KEY=your_key_here")
        return
    
    try:
        # Load IMDb data
        print("\n📂 Loading IMDb data...")
        imdb_basics = pd.read_parquet('data/processed/imdb_title_basics.parquet')
        imdb_ratings = pd.read_parquet('data/processed/imdb_title_ratings.parquet')
        
        # Merge basics with ratings
        imdb_df = imdb_basics.merge(imdb_ratings, on='tconst', how='inner')
        print(f"✅ Loaded {len(imdb_df):,} movies with ratings")
        
        # Filter for movies with sufficient votes (to reduce API calls)
        min_votes = 1000
        imdb_df = imdb_df[imdb_df['numVotes'] >= min_votes].copy()
        print(f"✅ Filtered to {len(imdb_df):,} movies with >={min_votes} votes")
        
        # Sort by number of votes (most popular first)
        imdb_df = imdb_df.sort_values('numVotes', ascending=False)
        
        # Initialize integrator
        integrator = TMDbIntegrator(api_key)
        
        # Check if --all flag is provided
        process_all = '--all' in sys.argv
        
        # Ask user for number of movies to process
        print(f"\n⚙️  Processing configuration:")
        print(f"   Total movies available: {len(imdb_df):,}")
        print(f"   TMDb API rate limit: ~50 requests/second")
        print(f"   Estimated time for 1000 movies: ~5 minutes")
        print(f"   Estimated time for all movies: ~{len(imdb_df)*0.25/60:.0f} minutes")
        
        if process_all:
            print(f"\n🚀 Processing ALL {len(imdb_df):,} movies (--all flag detected)")
            max_movies = None
        else:
            # Start with a sample for testing
            user_input = input(f"\nHow many movies to process? (default: 1000, 'all' for all movies): ")
            if user_input.lower() == 'all':
                max_movies = None
                print(f"🚀 Processing ALL {len(imdb_df):,} movies")
            else:
                max_movies = int(user_input or 1000)
        
        # Enrich data
        enriched_df = integrator.enrich_imdb_data(
            imdb_df, 
            max_requests=max_movies,
            rate_limit_delay=0.25
        )
        
        # Show statistics
        print("\n📊 TMDb Data Coverage:")
        stats = integrator.get_dataset_stats(enriched_df)
        print(f"   Total movies processed: {stats['total_movies']:,}")
        print(f"   TMDb matches found: {stats['with_tmdb_match']:,} ({stats['with_tmdb_match']/stats['total_movies']*100:.1f}%)")
        print(f"   Movies with budget data: {stats['with_budget']:,} ({stats['with_budget']/stats['total_movies']*100:.1f}%)")
        print(f"   Movies with revenue data: {stats['with_revenue']:,} ({stats['with_revenue']/stats['total_movies']*100:.1f}%)")
        print(f"   Movies with complete financial data: {stats['with_both_financial']:,} ({stats['with_both_financial']/stats['total_movies']*100:.1f}%)")
        
        if stats['with_budget'] > 0:
            print(f"\n💰 Financial Statistics:")
            print(f"   Average budget: ${stats['avg_budget']:,.0f}")
            print(f"   Average revenue: ${stats['avg_revenue']:,.0f}")
            print(f"   Total budget tracked: ${stats['total_budget']:,.0f}")
            print(f"   Total revenue tracked: ${stats['total_revenue']:,.0f}")
        
        # Save enriched data
        output_path = 'data/processed/imdb_tmdb_combined.parquet'
        enriched_df.to_parquet(output_path, index=False)
        print(f"\n💾 Saved enriched data to {output_path}")
        
        # Also save a CSV for easy viewing
        csv_path = 'data/processed/imdb_tmdb_combined.csv'
        enriched_df.to_csv(csv_path, index=False)
        print(f"💾 Saved CSV version to {csv_path}")
        
        print("\n🎉 TMDb integration completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Required data file not found - {e}")
        print("   Please run data_collection.py first to get IMDb data")
    except Exception as e:
        print(f"❌ Error during TMDb integration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
