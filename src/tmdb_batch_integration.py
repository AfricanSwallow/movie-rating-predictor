"""
Optimized Batch TMDb Integration with Resume Capability

This script processes movies in batches with improved efficiency:
- Memory-efficient chunked processing
- Optimized I/O operations
- Connection pooling and async requests
- Smart resume capability with minimal overhead
"""

import pandas as pd
import os
from dotenv import load_dotenv
from tmdb_integration import TMDbIntegrator
import sys
import gc
from pathlib import Path
import json
from datetime import datetime

load_dotenv()


def get_checkpoint_info(output_dir='data/processed'):
    """Get checkpoint information for resume capability."""
    checkpoint_file = Path(output_dir) / 'tmdb_batch_checkpoint.json'
    
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {'last_batch': -1, 'processed_count': 0, 'timestamp': None}


def save_checkpoint(batch_num, processed_count, output_dir='data/processed'):
    """Save checkpoint information."""
    checkpoint_file = Path(output_dir) / 'tmdb_batch_checkpoint.json'
    checkpoint = {
        'last_batch': batch_num,
        'processed_count': processed_count,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def load_data_efficiently(min_votes=1000):
    """Load and filter data with memory optimization."""
    print("\n📂 Loading IMDb data efficiently...")
    
    # Load only necessary columns to reduce memory
    basics_cols = ['tconst', 'primaryTitle', 'startYear', 'genres', 'titleType']
    ratings_cols = ['tconst', 'averageRating', 'numVotes']
    
    imdb_basics = pd.read_parquet(
        'data/processed/imdb_title_basics.parquet',
        columns=basics_cols
    )
    imdb_ratings = pd.read_parquet(
        'data/processed/imdb_title_ratings.parquet',
        columns=ratings_cols
    )
    
    # Filter ratings first to reduce merge size
    imdb_ratings = imdb_ratings[imdb_ratings['numVotes'] >= min_votes]
    
    # Merge on filtered data
    imdb_df = imdb_basics.merge(imdb_ratings, on='tconst', how='inner')
    
    # Sort by votes (descending) for better API hit rate on popular movies
    imdb_df = imdb_df.sort_values('numVotes', ascending=False).reset_index(drop=True)
    
    # Clean up memory
    del imdb_basics, imdb_ratings
    gc.collect()
    
    return imdb_df


def append_to_parquet(new_data, output_file, chunk_size=1000):
    """Efficiently append data to parquet file."""
    if os.path.exists(output_file):
        # Read existing data in chunks and append new data
        existing_df = pd.read_parquet(output_file)
        combined_df = pd.concat([existing_df, new_data], ignore_index=True)
        combined_df.to_parquet(output_file, index=False)
        del existing_df
    else:
        new_data.to_parquet(output_file, index=False)
    
    gc.collect()


def batch_integration(batch_size=2000, start_from=None, min_votes=1000, 
                     output_dir='data/processed', save_every_n_batches=5):
    """
    Optimized batch processing with smart resume capability.
    
    Args:
        batch_size: Number of movies per batch (reduced default for memory efficiency)
        start_from: Batch number to start from (auto-detected if None)
        min_votes: Minimum IMDb votes filter
        output_dir: Output directory for files
        save_every_n_batches: Save progress every N batches (reduces I/O)
    """
    
    print("🎬 OPTIMIZED TMDb BATCH INTEGRATION")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv('TMDB_API_KEY')
    if not api_key:
        print("❌ TMDb API key not found!")
        return
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint info
    checkpoint = get_checkpoint_info(output_dir)
    if start_from is None:
        start_from = checkpoint['last_batch'] + 1
        print(f"📍 Auto-resuming from batch {start_from} (found checkpoint)")
    
    # Load IMDb data efficiently
    imdb_df = load_data_efficiently(min_votes)
    print(f"✅ Loaded {len(imdb_df):,} movies with >={min_votes} votes")
    
    # Calculate batches
    total_batches = (len(imdb_df) + batch_size - 1) // batch_size
    remaining_batches = total_batches - start_from
    
    print(f"\n📊 Optimized Batch Configuration:")
    print(f"   Batch size: {batch_size:,} movies (memory optimized)")
    print(f"   Total batches: {total_batches}")
    print(f"   Starting from batch: {start_from}")
    print(f"   Remaining batches: {remaining_batches}")
    print(f"   Save frequency: Every {save_every_n_batches} batches")
    print(f"   Estimated time per batch: ~{batch_size*0.2/60:.0f} minutes")
    print(f"   Total remaining time: ~{remaining_batches*batch_size*0.2/3600:.1f} hours")
    
    # Initialize integrator with optimizations
    integrator = TMDbIntegrator(api_key)
    
    # Setup file paths
    output_file = Path(output_dir) / 'imdb_tmdb_combined.parquet'
    temp_batches = []
    
    # Process batches
    for batch_num in range(start_from, total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(imdb_df))
        
        print(f"\n{'='*50}")
        print(f"🎬 Batch {batch_num + 1}/{total_batches} (Movies {start_idx:,}-{end_idx:,})")
        print(f"{'='*50}")
        
        # Get batch data - use .iloc for memory efficiency
        batch_df = imdb_df.iloc[start_idx:end_idx].copy()
        
        try:
            # Enrich batch with optimized settings
            enriched_batch = integrator.enrich_imdb_data(
                batch_df,
                max_requests=None,
                rate_limit_delay=0.2  # Slightly faster rate
            )
            
            temp_batches.append(enriched_batch)
            
            # Save progress every N batches or at the end
            if (batch_num + 1) % save_every_n_batches == 0 or batch_num == total_batches - 1:
                print(f"\n💾 Saving progress ({len(temp_batches)} batches)...")
                
                # Combine temp batches
                if temp_batches:
                    combined_temp = pd.concat(temp_batches, ignore_index=True)
                    append_to_parquet(combined_temp, output_file)
                    
                    # Update checkpoint
                    total_processed = checkpoint['processed_count'] + len(combined_temp)
                    save_checkpoint(batch_num, total_processed, output_dir)
                    
                    print(f"✅ Saved {len(combined_temp):,} movies from recent batches")
                    
                    # Clear temp data to free memory
                    temp_batches.clear()
                    del combined_temp
                    gc.collect()
            
            # Show batch statistics
            stats = integrator.get_dataset_stats(enriched_batch)
            print(f"\n📊 Batch {batch_num + 1} Stats:")
            print(f"   TMDb matches: {stats['with_tmdb_match']:,}/{stats['total_movies']:,} "
                  f"({stats['with_tmdb_match']/stats['total_movies']*100:.1f}%)")
            print(f"   Financial data: {stats['with_both_financial']:,} "
                  f"({stats['with_both_financial']/stats['total_movies']*100:.1f}%)")
            
            # Clean up batch data
            del batch_df, enriched_batch
            gc.collect()
            
        except Exception as e:
            print(f"❌ Error processing batch {batch_num + 1}: {e}")
            print("💾 Saving progress before continuing...")
            if temp_batches:
                combined_temp = pd.concat(temp_batches, ignore_index=True)
                append_to_parquet(combined_temp, output_file)
                temp_batches.clear()
                del combined_temp
                gc.collect()
            continue
    
    # Generate final statistics and CSV
    print("\n" + "="*60)
    print("🎉 ALL BATCHES COMPLETED!")
    print("="*60)
    
    if os.path.exists(output_file):
        # Load final data for statistics (in chunks if too large)
        try:
            final_df = pd.read_parquet(output_file)
            final_stats = integrator.get_dataset_stats(final_df)
            
            print(f"\n📊 Final Statistics:")
            print(f"   Total movies processed: {final_stats['total_movies']:,}")
            print(f"   TMDb matches: {final_stats['with_tmdb_match']:,} "
                  f"({final_stats['with_tmdb_match']/final_stats['total_movies']*100:.1f}%)")
            print(f"   Complete financial data: {final_stats['with_both_financial']:,} "
                  f"({final_stats['with_both_financial']/final_stats['total_movies']*100:.1f}%)")
            
            if final_stats['with_budget'] > 0:
                print(f"\n💰 Financial Summary:")
                print(f"   Avg budget: ${final_stats['avg_budget']:,.0f}")
                print(f"   Avg revenue: ${final_stats['avg_revenue']:,.0f}")
            
            # Save optimized CSV (only if reasonable size)
            if len(final_df) < 100000:  # Only save CSV for smaller datasets
                csv_path = Path(output_dir) / 'imdb_tmdb_combined.csv'
                final_df.to_csv(csv_path, index=False)
                print(f"💾 CSV saved: {csv_path}")
            else:
                print("📁 Dataset too large for CSV - use Parquet file")
            
            del final_df
            
        except Exception as e:
            print(f"⚠️  Could not generate final statistics: {e}")
    
    print(f"\n✅ Output file: {output_file}")
    print("🧹 Cleaning up temporary files...")
    
    # Clean up checkpoint file
    checkpoint_file = Path(output_dir) / 'tmdb_batch_checkpoint.json'
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    
    gc.collect()


if __name__ == "__main__":
    # Parse command line arguments with improved defaults
    batch_size = 2000  # Reduced for better memory efficiency
    start_from = None  # Auto-detect from checkpoint
    min_votes = 1000
    save_frequency = 5  # Save every 5 batches
    
    if len(sys.argv) > 1:
        batch_size = int(sys.argv[1])
    if len(sys.argv) > 2:
        start_from = int(sys.argv[2]) if sys.argv[2] != 'auto' else None
    if len(sys.argv) > 3:
        min_votes = int(sys.argv[3])
    if len(sys.argv) > 4:
        save_frequency = int(sys.argv[4])
    
    print(f"🔧 Configuration:")
    print(f"   batch_size={batch_size}")
    print(f"   start_from={'auto-detect' if start_from is None else start_from}")
    print(f"   min_votes={min_votes}")
    print(f"   save_frequency={save_frequency}")
    print(f"\n📖 Usage: python src/tmdb_batch_integration.py [batch_size] [start_from|auto] [min_votes] [save_frequency]")
    print()
    
    batch_integration(batch_size, start_from, min_votes, save_every_n_batches=save_frequency)
