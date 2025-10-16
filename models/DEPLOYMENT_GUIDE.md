# Movie Profit Prediction Model - Deployment Guide

## Overview
This guide explains how to use the profit prediction model in a web application.

## Required Files
1. **`profit_prediction_model.pkl`** - The trained Linear Regression model
2. **`profit_prediction_metadata.json`** - Model metadata including feature list and performance metrics

## Model Information
- **Model Type**: Linear Regression
- **Target Variable**: Profit (USD)
- **Features**: 190 features total
- **Performance** (Test Set):
  - R² Score: 0.2613
  - RMSE: $261.6M
  - MAE: $187.6M

## Feature Breakdown
The model uses the following feature categories:

### 1. Genres (21 features)
Binary features for each genre:
- `genre_Action`, `genre_Adventure`, `genre_Animation`, `genre_Biography`, `genre_Comedy`, `genre_Crime`, `genre_Drama`, `genre_Family`, `genre_Fantasy`, `genre_Film-Noir`, `genre_History`, `genre_Horror`, `genre_Music`, `genre_Musical`, `genre_Mystery`, `genre_Romance`, `genre_Sci-Fi`, `genre_Sport`, `genre_Thriller`, `genre_War`, `genre_Western`

### 2. Directors (50 features)
Binary features for top 50 directors by IMDb ID:
- Format: `director_nm0000116` (James Cameron), `director_nm1349376` (Francis Lawrence), etc.
- See metadata file for complete list

### 3. Actors (50 features)
Binary features for top 50 actors by IMDb ID:
- Format: `actor_nm0695435` (Chris Pratt), `actor_nm0424060` (Ian McKellen), etc.
- See metadata file for complete list

### 4. Countries (21 features)
Binary features for top production countries:
- `country_US`, `country_GB`, `country_DE`, `country_FR`, `country_CA`, `country_AU`, `country_ES`, `country_JP`, `country_IT`, `country_CN`, `country_NZ`, `country_IN`, `country_HK`, `country_MX`, `country_IE`, `country_KR`, `country_BR`, `country_ZA`, `country_CH`, `country_BG`

### 5. Production Companies (31 features)
Binary features for top production companies:
- `company_Warner_Bros_Pictures`, `company_Universal_Pictures`, `company_Columbia_Pictures`, `company_20th_Century_Fox`, `company_Paramount_Pictures`, `company_Marvel_Studios`, etc.

### 6. Languages (12 features)
Binary features for top languages:
- `lang_en`, `lang_ja`, `lang_it`, `lang_fr`, `lang_ko`, `lang_es`, `lang_de`, `lang_pt`, `lang_hi`, `lang_da`, `lang_zh`, `lang_fa`

### 7. Other Features (5 features)
- `isAdult` - Binary (0 or 1)
- `budget_millions` - Numerical (budget in millions)
- `runtime` - Numerical (runtime in minutes)
- `year` - Numerical (release year)
- `genre_count` - Numerical (number of genres)
- `country_count` - Numerical (number of production countries)
- `company_count` - Numerical (number of production companies)

## Usage Example (Python)

```python
import joblib
import json
import numpy as np
import pandas as pd

# Load the model and metadata
model = joblib.load('profit_prediction_model.pkl')
with open('profit_prediction_metadata.json', 'r') as f:
    metadata = json.load(f)

# Get feature list (190 features in correct order)
feature_columns = metadata['feature_columns']

# Example: Create input for a new movie
# Initialize all features to 0
input_data = {feature: 0 for feature in feature_columns}

# Set feature values based on user selections
input_data['genre_Action'] = 1
input_data['genre_Adventure'] = 1
input_data['genre_Sci-Fi'] = 1
input_data['director_nm0000116'] = 1  # James Cameron
input_data['actor_nm0695435'] = 1      # Chris Pratt
input_data['country_US'] = 1
input_data['company_Marvel_Studios'] = 1
input_data['lang_en'] = 1
input_data['budget_millions'] = 200.0  # $200M budget
input_data['runtime'] = 150            # 150 minutes
input_data['year'] = 2025
input_data['genre_count'] = 3
input_data['country_count'] = 1
input_data['company_count'] = 1
input_data['isAdult'] = 0

# Convert to DataFrame (required for sklearn)
input_df = pd.DataFrame([input_data])

# Ensure correct order
input_df = input_df[feature_columns]

# Make prediction
predicted_profit = model.predict(input_df)[0]

print(f"Predicted Profit: ${predicted_profit:,.0f}")
print(f"Predicted Profit (Millions): ${predicted_profit/1e6:.2f}M")
```

## Web App Implementation Tips

### 1. **User Interface**
Create form inputs for:
- **Genres**: Multi-select checkboxes
- **Directors**: Dropdown (with name mapping - see Director/Actor Mapping section)
- **Actors**: Dropdown (with name mapping)
- **Countries**: Multi-select
- **Companies**: Multi-select
- **Languages**: Dropdown
- **Budget**: Number input (millions)
- **Runtime**: Number input (minutes)
- **Year**: Number input
- **Is Adult**: Checkbox

### 2. **Backend Processing**
```python
from flask import Flask, request, jsonify
import joblib
import json
import pandas as pd

app = Flask(__name__)

# Load model at startup
model = joblib.load('profit_prediction_model.pkl')
with open('profit_prediction_metadata.json', 'r') as f:
    metadata = json.load(f)
feature_columns = metadata['feature_columns']

@app.route('/predict', methods=['POST'])
def predict_profit():
    data = request.json
    
    # Initialize all features to 0
    input_data = {feature: 0 for feature in feature_columns}
    
    # Set genres
    for genre in data.get('genres', []):
        input_data[f'genre_{genre}'] = 1
    
    # Set directors
    for director_id in data.get('directors', []):
        input_data[f'director_{director_id}'] = 1
    
    # Set actors
    for actor_id in data.get('actors', []):
        input_data[f'actor_{actor_id}'] = 1
    
    # Set countries
    for country in data.get('countries', []):
        input_data[f'country_{country}'] = 1
    
    # Set companies
    for company in data.get('companies', []):
        company_key = company.replace(' ', '_').replace('.', '').replace(',', '')[:30]
        input_data[f'company_{company_key}'] = 1
    
    # Set language
    if 'language' in data:
        input_data[f"lang_{data['language']}"] = 1
    
    # Set numerical features
    input_data['budget_millions'] = data.get('budget', 100)
    input_data['runtime'] = data.get('runtime', 120)
    input_data['year'] = data.get('year', 2025)
    input_data['isAdult'] = data.get('isAdult', 0)
    
    # Set counts
    input_data['genre_count'] = len(data.get('genres', []))
    input_data['country_count'] = len(data.get('countries', []))
    input_data['company_count'] = len(data.get('companies', []))
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])[feature_columns]
    
    # Predict
    predicted_profit = float(model.predict(input_df)[0])
    
    return jsonify({
        'predicted_profit': predicted_profit,
        'predicted_profit_millions': predicted_profit / 1e6,
        'formatted': f"${predicted_profit:,.0f}"
    })

if __name__ == '__main__':
    app.run(debug=True)
```

## Director/Actor Mapping

The model uses IMDb IDs (e.g., `nm0000116`) for directors and actors. You'll need to provide a name mapping in your web app.

### Top 10 Directors by Impact:
| IMDb ID | Name | Impact |
|---------|------|--------|
| nm0000116 | James Cameron | +$773M |
| nm1349376 | Francis Lawrence | +$324M |
| nm0796117 | M. Night Shyamalan | +$313M |
| nm0001060 | Chris Columbus | +$270M |
| nm0680846 | Todd Phillips | +$243M |
| nm0751648 | Joe Russo | +$225M |
| nm0751577 | Anthony Russo | +$225M |
| nm0001392 | David Yates | +$201M |
| nm0004716 | James Wan | +$191M |
| nm0811583 | Sam Mendes | +$179M |

### Top 10 Actors by Impact:
| IMDb ID | Name | Impact |
|---------|------|--------|
| nm0695435 | Chris Pratt | +$281M |
| nm0424060 | Ian McKellen | +$198M |
| nm0000226 | Will Smith | +$176M |
| nm0000437 | Jeremy Renner | +$169M |
| nm0262635 | Willem Dafoe | +$130M |
| nm0000148 | Harrison Ford | +$124M |
| nm0005212 | Andy Serkis | +$124M |
| nm0001570 | Leonardo DiCaprio | +$121M |
| nm0914612 | Emma Watson | +$119M |
| nm0736622 | Laurence Fishburne | +$88M |

**To get complete mapping**: You'll need to download IMDb's `name.basics.tsv.gz` file and create a lookup dictionary, or provide a pre-built JSON mapping file with all 100 directors and actors.

## Important Notes

1. **Feature Order**: The features MUST be in the exact order specified in `feature_columns` from the metadata file.

2. **Missing Features**: If a user doesn't select a particular feature, set it to 0 (for binary features) or appropriate default (for numerical features).

3. **Budget Scale**: The model expects budget in millions. If user enters $150,000,000, convert to 150.

4. **Validation**: Validate user inputs:
   - Budget > 0
   - Runtime > 0
   - Year between 1900-2100
   - Valid genre/country/company/language selections

5. **Model Limitations**:
   - R² of 0.26 means the model explains 26% of variance
   - Predictions are estimates with ±$262M average error
   - Best suited for mainstream movies with known directors/actors

## Testing the Model

```python
# Test with a known movie profile
test_cases = [
    {
        "name": "High-budget Action with James Cameron",
        "genres": ["Action", "Sci-Fi", "Adventure"],
        "directors": ["nm0000116"],  # James Cameron
        "actors": ["nm0695435"],      # Chris Pratt
        "countries": ["US"],
        "companies": ["20th Century Fox"],
        "language": "en",
        "budget": 300,  # $300M
        "runtime": 162,
        "year": 2025,
        "expected_profit": "High (>$500M)"
    },
    {
        "name": "Low-budget Drama",
        "genres": ["Drama"],
        "directors": [],
        "actors": [],
        "countries": ["US"],
        "companies": ["Miramax"],
        "language": "en",
        "budget": 5,  # $5M
        "runtime": 95,
        "year": 2025,
        "expected_profit": "Low (<$50M)"
    }
]
```

## Support

For questions or issues:
1. Check that all 190 features are present in input
2. Verify feature values are in correct format (0/1 for binary, numerical for others)
3. Ensure feature order matches metadata file

## License & Attribution

Please credit the original model creators when using this model in your web application.
