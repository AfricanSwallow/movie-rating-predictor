"""
Quick test script to verify TMDb API integration and check a specific movie.
"""

import os
import sys
from dotenv import load_dotenv
import requests

load_dotenv()


def test_tmdb_api():
    """Test TMDb API connection and show example data."""
    
    api_key = os.getenv('TMDB_API_KEY')
    
    if not api_key:
        print("❌ TMDb API key not found!")
        print("\n📝 Setup instructions:")
        print("   1. Get API key at https://www.themoviedb.org/settings/api")
        print("   2. Create .env file: cp .env.example .env")
        print("   3. Add your key: TMDB_API_KEY=your_key_here")
        return False
    
    print("✅ TMDb API key found")
    print(f"   Key: {api_key[:10]}...{api_key[-4:]}\n")
    
    # Test with The Shawshank Redemption (tt0111161)
    test_movies = [
        {'imdb_id': 'tt0111161', 'title': 'The Shawshank Redemption'},
        {'imdb_id': 'tt0068646', 'title': 'The Godfather'},
        {'imdb_id': 'tt0468569', 'title': 'The Dark Knight'}
    ]
    
    print("🎬 Testing API with sample movies:\n")
    
    for movie in test_movies:
        imdb_id = movie['imdb_id']
        expected_title = movie['title']
        
        try:
            # Find movie by IMDb ID
            url = f"https://api.themoviedb.org/3/find/{imdb_id}"
            params = {
                'api_key': api_key,
                'external_source': 'imdb_id'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            movie_results = data.get('movie_results', [])
            
            if not movie_results:
                print(f"❌ {expected_title} ({imdb_id}): Not found")
                continue
            
            # Get first result
            result = movie_results[0]
            tmdb_id = result['id']
            
            # Get detailed info
            detail_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
            detail_params = {'api_key': api_key}
            
            detail_response = requests.get(detail_url, params=detail_params, timeout=10)
            detail_response.raise_for_status()
            
            details = detail_response.json()
            
            # Display results
            print(f"✅ {expected_title} ({imdb_id})")
            print(f"   TMDb ID: {details.get('id')}")
            print(f"   Title: {details.get('title')}")
            print(f"   Year: {details.get('release_date', 'N/A')[:4]}")
            print(f"   Budget: ${details.get('budget', 0):,}")
            print(f"   Revenue: ${details.get('revenue', 0):,}")
            
            if details.get('budget', 0) > 0 and details.get('revenue', 0) > 0:
                roi = ((details['revenue'] - details['budget']) / details['budget']) * 100
                print(f"   ROI: {roi:.1f}%")
            
            print(f"   TMDb Rating: {details.get('vote_average')}/10")
            print(f"   Popularity: {details.get('popularity')}")
            
            companies = details.get('production_companies', [])
            if companies:
                company_names = [c['name'] for c in companies[:3]]
                print(f"   Production: {', '.join(company_names)}")
            
            print()
            
        except requests.RequestException as e:
            print(f"❌ Error fetching {expected_title}: {e}\n")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}\n")
            return False
    
    print("🎉 TMDb API test successful!")
    print("\n📊 You can now run:")
    print("   python src/tmdb_integration.py")
    print("   or open notebooks/05_tmdb_financial_analysis.ipynb")
    
    return True


def lookup_movie(imdb_id: str):
    """Look up a specific movie by IMDb ID."""
    
    api_key = os.getenv('TMDB_API_KEY')
    
    if not api_key:
        print("❌ TMDb API key not found in .env file")
        return
    
    try:
        # Find movie
        url = f"https://api.themoviedb.org/3/find/{imdb_id}"
        params = {
            'api_key': api_key,
            'external_source': 'imdb_id'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        movie_results = data.get('movie_results', [])
        
        if not movie_results:
            print(f"❌ Movie not found: {imdb_id}")
            return
        
        result = movie_results[0]
        tmdb_id = result['id']
        
        # Get details
        detail_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        detail_params = {'api_key': api_key}
        
        detail_response = requests.get(detail_url, params=detail_params, timeout=10)
        detail_response.raise_for_status()
        
        details = detail_response.json()
        
        # Print formatted output
        print(f"\n🎬 {details.get('title')} ({details.get('release_date', 'N/A')[:4]})")
        print("=" * 60)
        print(f"IMDb ID: {imdb_id}")
        print(f"TMDb ID: {details.get('id')}")
        print(f"Tagline: {details.get('tagline', 'N/A')}")
        print(f"\n💰 FINANCIAL DATA:")
        print(f"   Budget: ${details.get('budget', 0):,}")
        print(f"   Revenue: ${details.get('revenue', 0):,}")
        
        if details.get('budget', 0) > 0 and details.get('revenue', 0) > 0:
            profit = details['revenue'] - details['budget']
            roi = (profit / details['budget']) * 100
            print(f"   Profit: ${profit:,}")
            print(f"   ROI: {roi:.1f}%")
        
        print(f"\n⭐ RATINGS:")
        print(f"   TMDb: {details.get('vote_average')}/10 ({details.get('vote_count'):,} votes)")
        print(f"   Popularity: {details.get('popularity')}")
        
        print(f"\n📊 METADATA:")
        print(f"   Runtime: {details.get('runtime')} minutes")
        print(f"   Language: {details.get('original_language')}")
        
        genres = details.get('genres', [])
        if genres:
            genre_names = [g['name'] for g in genres]
            print(f"   Genres: {', '.join(genre_names)}")
        
        companies = details.get('production_companies', [])
        if companies:
            company_names = [c['name'] for c in companies]
            print(f"   Production: {', '.join(company_names)}")
        
        countries = details.get('production_countries', [])
        if countries:
            country_names = [c['name'] for c in countries]
            print(f"   Countries: {', '.join(country_names)}")
        
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Look up specific movie
        imdb_id = sys.argv[1]
        if not imdb_id.startswith('tt'):
            imdb_id = 'tt' + imdb_id
        lookup_movie(imdb_id)
    else:
        # Run test
        test_tmdb_api()
