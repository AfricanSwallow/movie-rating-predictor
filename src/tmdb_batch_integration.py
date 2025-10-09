"""
Batch TMDb Integration with Resume Capability

This script processes movies in batches and can resume from where it left off.
"""

import pandas as pd
import os
from dotenv import load_dotenv
from tmdb_integration import TMDbIntegrator
import sys

load_dotenv()


def batch_integration(batch_size=5000, start_from=0, min_votes=1000):
    """
    Process TMDb integration in batches with resume capability.
    
    Args:
        batch_size: Number of movies per batch
        start_from: Batch number to start from (0-indexed)
        min_votes: Minimum IMDb votes filter
    """
    
    print("🎬 TMDb BATCH INTEGRATION")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv('TMDB_API_KEY')
    if not api_key:
        print("❌ TMDb API key not found!")
        return
    
    # Load IMDb data
    print("\n📂 Loading IMDb data...")
    imdb_basics = pd.read_parquet('data/processed/imdb_title_basics.parquet')
    imdb_ratings = pd.read_parquet('data/processed/imdb_title_ratings.parquet')
    
    # Merge and filter
    imdb_df = imdb_basics.merge(imdb_ratings, on='tconst', how='inner')
    imdb_df = imdb_df[imdb_df['numVotes'] >= min_votes].copy()
    imdb_df = imdb_df.sort_values('numVotes', ascending=False).reset_index(drop=True)
    
    print(f"✅ Loaded {len(imdb_df):,} movies with >={min_votes} votes")
    
    # Calculate batches
    total_batches = (len(imdb_df) + batch_size - 1) // batch_size
    print(f"\n📊 Batch Configuration:")
    print(f"   Batch size: {batch_size:,} movies")
    print(f"   Total batches: {total_batches}")
    print(f"   Starting from batch: {start_from}")
    print(f"   Estimated time per batch: ~{batch_size*0.25/60:.0f} minutes")
    print(f"   Total estimated time: ~{len(imdb_df)*0.25/3600:.1f} hours")
    
    # Initialize integrator
    integrator = TMDbIntegrator(api_key)
    
    # Check for existing data
    output_file = 'data/processed/imdb_tmdb_combined.parquet'
    if os.path.exists(output_file) and start_from > 0:
        print(f"\n📂 Loading existing data from {output_file}")
        existing_df = pd.read_parquet(output_file)
        print(f"   Found {len(existing_df):,} already processed movies")
    else:
        existing_df = None
    
    # Process batches
    all_results = []
    
    for batch_num in range(start_from, total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(imdb_df))
        
        print(f"\n{'='*60}")
        print(f"🎬 Processing Batch {batch_num + 1}/{total_batches}")
        print(f"   Movies {start_idx:,} to {end_idx:,}")
        print(f"{'='*60}")
        
        batch_df = imdb_df.iloc[start_idx:end_idx].copy()
        
        # Enrich batch
        enriched_batch = integrator.enrich_imdb_data(
            batch_df,
            max_requests=None,  # Process entire batch
            rate_limit_delay=0.25
        )
        
        all_results.append(enriched_batch)
        
        # Save intermediate results
        if existing_df is not None:
            combined_df = pd.concat([existing_df] + all_results, ignore_index=True)
        else:
            combined_df = pd.concat(all_results, ignore_index=True)
        
        combined_df.to_parquet(output_file, index=False)
        print(f"\n💾 Saved progress: {len(combined_df):,} movies processed")
        
        # Show statistics for this batch
        stats = integrator.get_dataset_stats(enriched_batch)
        print(f"\n📊 Batch {batch_num + 1} Statistics:")
        print(f"   TMDb matches: {stats['with_tmdb_match']:,} ({stats['with_tmdb_match']/stats['total_movies']*100:.1f}%)")
        print(f"   With budget: {stats['with_budget']:,} ({stats['with_budget']/stats['total_movies']*100:.1f}%)")
        print(f"   With revenue: {stats['with_revenue']:,} ({stats['with_revenue']/stats['total_movies']*100:.1f}%)")
        
    # Final statistics
    print("\n" + "="*60)
    print("🎉 ALL BATCHES COMPLETED!")
    print("="*60)
    
    final_stats = integrator.get_dataset_stats(combined_df)
    print(f"\n📊 Final Statistics:")
    print(f"   Total movies processed: {final_stats['total_movies']:,}")
    print(f"   TMDb matches found: {final_stats['with_tmdb_match']:,} ({final_stats['with_tmdb_match']/final_stats['total_movies']*100:.1f}%)")
    print(f"   Movies with budget data: {final_stats['with_budget']:,} ({final_stats['with_budget']/final_stats['total_movies']*100:.1f}%)")
    print(f"   Movies with revenue data: {final_stats['with_revenue']:,} ({final_stats['with_revenue']/final_stats['total_movies']*100:.1f}%)")
    print(f"   Movies with complete financial data: {final_stats['with_both_financial']:,} ({final_stats['with_both_financial']/final_stats['total_movies']*100:.1f}%)")
    
    if final_stats['with_budget'] > 0:
        print(f"\n💰 Financial Statistics:")
        print(f"   Average budget: ${final_stats['avg_budget']:,.0f}")
        print(f"   Average revenue: ${final_stats['avg_revenue']:,.0f}")
        print(f"   Total budget tracked: ${final_stats['total_budget']:,.0f}")
        print(f"   Total revenue tracked: ${final_stats['total_revenue']:,.0f}")
    
    # Save CSV
    csv_path = 'data/processed/imdb_tmdb_combined.csv'
    combined_df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved CSV version to {csv_path}")
    
    print(f"\n✅ Output file: {output_file}")


if __name__ == "__main__":
    # Parse command line arguments
    batch_size = 5000
    start_from = 0
    min_votes = 1000
    
    if len(sys.argv) > 1:
        batch_size = int(sys.argv[1])
    if len(sys.argv) > 2:
        start_from = int(sys.argv[2])
    if len(sys.argv) > 3:
        min_votes = int(sys.argv[3])
    
    print(f"Configuration: batch_size={batch_size}, start_from={start_from}, min_votes={min_votes}")
    print("Usage: python src/tmdb_batch_integration.py [batch_size] [start_from] [min_votes]")
    print()
    
    batch_integration(batch_size, start_from, min_votes)
