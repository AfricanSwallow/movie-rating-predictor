# TMDb Integration Guide

This guide explains how to integrate financial data from The Movie Database (TMDb) with your IMDb dataset.

## Overview

**IMDb Dataset** provides:
- Movie titles, years, genres
- IMDb ratings and vote counts
- Cast and crew information
- Runtime

**TMDb Dataset** provides:
- ✅ **Budget** (production costs)
- ✅ **Revenue** (box office earnings)
- ✅ **Production companies**
- ✅ **Production countries**
- ✅ **TMDb popularity scores**
- ✅ **Additional ratings**

## Getting Started

### 1. Get a TMDb API Key

1. Create a free account at [themoviedb.org/signup](https://www.themoviedb.org/signup)
2. Go to [API Settings](https://www.themoviedb.org/settings/api)
3. Request an API key (free for non-commercial/educational use)
4. Copy your API key

### 2. Configure API Key

Create a `.env` file in the project root:

```bash
TMDB_API_KEY=your_api_key_here
```

### 3. Run the Integration

```bash
# Option 1: Run the integration script
python src/tmdb_integration.py

# Option 2: Use the Jupyter notebook
jupyter notebook notebooks/05_tmdb_financial_analysis.ipynb
```

## How It Works

The integration uses IMDb IDs as the joining key:

```
IMDb Data (tconst) → TMDb API (find by IMDb ID) → TMDb Movie Details
```

For each movie in your IMDb dataset:
1. Query TMDb API using the IMDb ID (`tt0111161`)
2. Fetch complete movie details including financial data
3. Extract and join with IMDb data

## Output

The integration creates: `data/processed/imdb_tmdb_combined.parquet`

**New columns added:**
- `tmdb_id` - TMDb movie ID
- `budget` - Production budget (USD)
- `revenue` - Box office revenue (USD)
- `roi` - Return on Investment (%)
- `profit` - Revenue - Budget
- `tmdb_popularity` - TMDb popularity score
- `tmdb_vote_average` - TMDb user rating
- `tmdb_vote_count` - Number of TMDb votes
- `production_countries` - Country codes (comma-separated)
- `production_companies` - Production company names
- `has_budget_data` - Boolean flag
- `has_revenue_data` - Boolean flag

## API Rate Limits

- TMDb allows ~50 requests per second
- Processing 1,000 movies takes ~5 minutes
- Processing 10,000 movies takes ~50 minutes

The script includes automatic rate limiting (0.25 seconds between requests).

## Data Coverage

Financial data availability varies:
- **Recent movies (2000+)**: ~80-90% coverage
- **1990s movies**: ~50-70% coverage  
- **Older movies**: ~20-40% coverage
- **Independent/international films**: Lower coverage

Not all movies have budget/revenue data in TMDb. The integration script filters for movies with sufficient votes to maximize data coverage.

## Example Usage

```python
import pandas as pd

# Load integrated data
df = pd.read_parquet('data/processed/imdb_tmdb_combined.parquet')

# Filter for movies with complete financial data
df_financial = df[(df['budget'] > 0) & (df['revenue'] > 0)]

# Calculate average ROI by genre
genre_columns = [col for col in df.columns if col.startswith('genre_')]
for genre in genre_columns:
    genre_name = genre.replace('genre_', '')
    genre_movies = df_financial[df_financial[genre] == 1]
    if len(genre_movies) > 0:
        avg_roi = genre_movies['roi'].mean()
        print(f"{genre_name}: {avg_roi:.1f}% ROI")

# Analyze budget vs rating
import matplotlib.pyplot as plt
plt.scatter(df_financial['budget']/1e6, df_financial['averageRating'])
plt.xlabel('Budget (Millions)')
plt.ylabel('IMDb Rating')
plt.show()
```

## New Features You Can Create

With financial data, you can engineer features like:

1. **Budget Categories**: 
   - Low budget (<$5M)
   - Medium budget ($5M-$50M)
   - High budget (>$50M)

2. **Financial Performance**:
   - ROI (Return on Investment)
   - Profit margin
   - Budget per minute of runtime

3. **Production Analysis**:
   - Major studio vs independent
   - Country of origin effects
   - Co-production patterns

4. **Log-transformed values**:
   - `log_budget` = log(budget)
   - `log_revenue` = log(revenue)
   - Better for machine learning models

## Troubleshooting

**"TMDb API key not found"**
- Check your `.env` file exists in project root
- Verify the key is: `TMDB_API_KEY=...`
- Try restarting your Python environment

**"Rate limit exceeded"**
- The script automatically handles rate limiting
- If you hit limits, wait a few seconds and retry
- Consider processing in smaller batches

**"Low match rate"**
- TMDb has better coverage for popular/recent movies
- Filter IMDb data for movies with more votes
- Some IMDb IDs might not exist in TMDb

## Next Steps

1. **Explore the notebook**: `notebooks/05_tmdb_financial_analysis.ipynb`
2. **Update your models**: Add financial features to your prediction models
3. **Build new models**: Predict revenue or ROI based on ratings
4. **Analyze patterns**: Discover what makes movies financially successful

## Resources

- [TMDb API Documentation](https://developers.themoviedb.org/3)
- [IMDb Dataset Documentation](https://www.imdb.com/interfaces/)
- Example notebook: `notebooks/05_tmdb_financial_analysis.ipynb`
