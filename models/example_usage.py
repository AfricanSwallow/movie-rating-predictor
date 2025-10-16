"""
Example script showing how to use the profit prediction model
"""

import joblib
import json
import pandas as pd

def load_model():
    """Load the model and metadata"""
    model = joblib.load('profit_prediction_model.pkl')
    with open('profit_prediction_metadata.json', 'r') as f:
        metadata = json.load(f)
    return model, metadata

def predict_profit(model, feature_columns, movie_data):
    """
    Predict profit for a movie
    
    Args:
        model: Loaded sklearn model
        feature_columns: List of feature names in correct order
        movie_data: Dictionary with movie characteristics
        
    Returns:
        Predicted profit in USD
    """
    # Initialize all features to 0
    input_data = {feature: 0 for feature in feature_columns}
    
    # Set genres
    for genre in movie_data.get('genres', []):
        input_data[f'genre_{genre}'] = 1
    
    # Set directors (IMDb IDs)
    for director_id in movie_data.get('directors', []):
        input_data[f'director_{director_id}'] = 1
    
    # Set actors (IMDb IDs)
    for actor_id in movie_data.get('actors', []):
        input_data[f'actor_{actor_id}'] = 1
    
    # Set countries
    for country in movie_data.get('countries', []):
        input_data[f'country_{country}'] = 1
    
    # Set companies
    for company in movie_data.get('companies', []):
        company_key = company.replace(' ', '_').replace('.', '').replace(',', '')[:30]
        input_data[f'company_{company_key}'] = 1
    
    # Set language
    if 'language' in movie_data:
        input_data[f"lang_{movie_data['language']}"] = 1
    
    # Set numerical features
    input_data['budget_millions'] = movie_data.get('budget_millions', 100)
    input_data['runtime'] = movie_data.get('runtime', 120)
    input_data['year'] = movie_data.get('year', 2025)
    input_data['isAdult'] = movie_data.get('isAdult', 0)
    
    # Set counts
    input_data['genre_count'] = len(movie_data.get('genres', []))
    input_data['country_count'] = len(movie_data.get('countries', []))
    input_data['company_count'] = len(movie_data.get('companies', []))
    
    # Convert to DataFrame with correct order
    input_df = pd.DataFrame([input_data])[feature_columns]
    
    # Predict
    predicted_profit = float(model.predict(input_df)[0])
    
    return predicted_profit

def main():
    """Run example predictions"""
    print("Loading model...")
    model, metadata = load_model()
    feature_columns = metadata['feature_columns']
    print(f"✓ Model loaded with {len(feature_columns)} features\n")
    
    # Example 1: High-budget blockbuster with James Cameron
    print("="*80)
    print("Example 1: High-Budget Sci-Fi Action Movie")
    print("="*80)
    movie1 = {
        'genres': ['Action', 'Sci-Fi', 'Adventure'],
        'directors': ['nm0000116'],  # James Cameron
        'actors': ['nm0695435', 'nm0424060'],  # Chris Pratt, Ian McKellen
        'countries': ['US'],
        'companies': ['20th Century Fox', 'Lucasfilm Ltd'],
        'language': 'en',
        'budget_millions': 300,
        'runtime': 162,
        'year': 2025,
        'isAdult': 0
    }
    
    profit1 = predict_profit(model, feature_columns, movie1)
    print(f"Input:")
    print(f"  Genres: {', '.join(movie1['genres'])}")
    print(f"  Director: James Cameron")
    print(f"  Actors: Chris Pratt, Ian McKellen")
    print(f"  Budget: ${movie1['budget_millions']}M")
    print(f"  Runtime: {movie1['runtime']} min")
    print(f"\nPredicted Profit: ${profit1:,.0f}")
    print(f"Predicted Profit: ${profit1/1e6:.2f}M")
    print(f"Expected ROI: {(profit1 / (movie1['budget_millions'] * 1e6)) * 100:.1f}%\n")
    
    # Example 2: Mid-budget comedy
    print("="*80)
    print("Example 2: Mid-Budget Comedy")
    print("="*80)
    movie2 = {
        'genres': ['Comedy', 'Romance'],
        'directors': [],  # No top director
        'actors': [],  # No top actor
        'countries': ['US'],
        'companies': ['Universal Pictures'],
        'language': 'en',
        'budget_millions': 30,
        'runtime': 98,
        'year': 2025,
        'isAdult': 0
    }
    
    profit2 = predict_profit(model, feature_columns, movie2)
    print(f"Input:")
    print(f"  Genres: {', '.join(movie2['genres'])}")
    print(f"  Director: Unknown")
    print(f"  Actors: Unknown")
    print(f"  Budget: ${movie2['budget_millions']}M")
    print(f"  Runtime: {movie2['runtime']} min")
    print(f"\nPredicted Profit: ${profit2:,.0f}")
    print(f"Predicted Profit: ${profit2/1e6:.2f}M")
    print(f"Expected ROI: {(profit2 / (movie2['budget_millions'] * 1e6)) * 100:.1f}%\n")
    
    # Example 3: Low-budget horror
    print("="*80)
    print("Example 3: Low-Budget Horror")
    print("="*80)
    movie3 = {
        'genres': ['Horror', 'Thriller'],
        'directors': [],
        'actors': [],
        'countries': ['US'],
        'companies': ['Blumhouse Productions'],  # Not in top 31
        'language': 'en',
        'budget_millions': 5,
        'runtime': 92,
        'year': 2025,
        'isAdult': 0
    }
    
    profit3 = predict_profit(model, feature_columns, movie3)
    print(f"Input:")
    print(f"  Genres: {', '.join(movie3['genres'])}")
    print(f"  Director: Unknown")
    print(f"  Actors: Unknown")
    print(f"  Budget: ${movie3['budget_millions']}M")
    print(f"  Runtime: {movie3['runtime']} min")
    print(f"\nPredicted Profit: ${profit3:,.0f}")
    print(f"Predicted Profit: ${profit3/1e6:.2f}M")
    print(f"Expected ROI: {(profit3 / (movie3['budget_millions'] * 1e6)) * 100:.1f}%\n")
    
    print("="*80)
    print("Model Performance (Test Set):")
    print("="*80)
    print(f"R² Score: {metadata['performance_metrics']['test']['r2']:.4f}")
    print(f"RMSE: ${metadata['performance_metrics']['test']['rmse']:,.0f}")
    print(f"MAE: ${metadata['performance_metrics']['test']['mae']:,.0f}")

if __name__ == "__main__":
    main()
