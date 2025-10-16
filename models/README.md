# Movie Profit Prediction Model - Deployment Package

## 📦 Package Contents

This folder contains everything needed to deploy the movie profit prediction model:

### Required Files for Deployment
1. ✅ **profit_prediction_model.pkl** (3.5 KB) - The trained Linear Regression model
2. ✅ **profit_prediction_metadata.json** (10 KB) - Model configuration and feature list
3. ✅ **name_mappings.json** (3 KB) - Director/Actor/Company name mappings for UI

### Documentation & Examples
4. 📖 **DEPLOYMENT_GUIDE.md** - Complete integration guide with code examples
5. 🐍 **example_usage.py** - Working Python example script
6. 📄 **README.md** - This file

## 🚀 Quick Start

### Step 1: Copy Files to Your Project
```bash
# Copy these 3 files to your web app directory:
- profit_prediction_model.pkl
- profit_prediction_metadata.json
- name_mappings.json
```

### Step 2: Install Dependencies
```bash
pip install scikit-learn pandas numpy joblib
```

### Step 3: Test the Model
```bash
python example_usage.py
```

Expected output:
```
Loading model...
✓ Model loaded with 190 features

================================================================================
Example 1: High-Budget Sci-Fi Action Movie
================================================================================
...
Predicted Profit: $773,467,226
```

## 📊 Model Overview

- **Type**: Linear Regression
- **Features**: 190 total
  - 21 genres (Action, Comedy, Drama, etc.)
  - 50 directors (James Cameron, Francis Lawrence, etc.)
  - 50 actors (Chris Pratt, Ian McKellen, etc.)
  - 21 countries (US, GB, FR, etc.)
  - 31 production companies (Warner Bros, Marvel, etc.)
  - 12 languages (en, ja, fr, etc.)
  - 5 numerical features (budget, runtime, year, counts)

- **Performance** (Test Set):
  - R² = 0.26 (explains 26% of variance)
  - RMSE = $261.6M
  - MAE = $187.6M

## 🎯 Top Predictive Features

### Directors with Highest Impact
| Director | Impact |
|----------|--------|
| James Cameron | +$773M |
| Francis Lawrence | +$324M |
| M. Night Shyamalan | +$313M |

### Actors with Highest Impact
| Actor | Impact |
|-------|--------|
| Chris Pratt | +$281M |
| Ian McKellen | +$198M |
| Will Smith | +$176M |

### Companies with Highest Impact
| Company | Impact |
|---------|--------|
| Lucasfilm Ltd | +$277M |
| Marvel Studios | +$200M+ |

## 💻 Usage Example

```python
import joblib
import json
import pandas as pd

# Load model
model = joblib.load('profit_prediction_model.pkl')
with open('profit_prediction_metadata.json', 'r') as f:
    metadata = json.load(f)

# Prepare input (190 features, all zeros by default)
features = {col: 0 for col in metadata['feature_columns']}

# Set user selections
features['genre_Action'] = 1
features['genre_Sci-Fi'] = 1
features['director_nm0000116'] = 1  # James Cameron
features['actor_nm0695435'] = 1      # Chris Pratt
features['country_US'] = 1
features['company_Marvel_Studios'] = 1
features['lang_en'] = 1
features['budget_millions'] = 200
features['runtime'] = 150
features['year'] = 2025
features['genre_count'] = 2
features['country_count'] = 1
features['company_count'] = 1

# Predict
input_df = pd.DataFrame([features])[metadata['feature_columns']]
profit = model.predict(input_df)[0]

print(f"Predicted Profit: ${profit:,.0f}")
# Output: Predicted Profit: $XXX,XXX,XXX
```

## 🌐 Web App Integration

### Frontend (User Selections)
```javascript
{
  "genres": ["Action", "Sci-Fi"],
  "directors": ["nm0000116"],  // James Cameron
  "actors": ["nm0695435"],      // Chris Pratt
  "countries": ["US"],
  "companies": ["Marvel Studios"],
  "language": "en",
  "budget": 200,               // millions
  "runtime": 150,              // minutes
  "year": 2025,
  "isAdult": false
}
```

### Backend (Flask API)
```python
@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.json
    # Process input (see example_usage.py)
    profit = predict_profit(model, features, user_input)
    return jsonify({'profit': profit})
```

## ⚠️ Important Notes

1. **Feature Order Matters**: Features MUST be in the exact order from `metadata['feature_columns']`

2. **Budget Scale**: Model expects budget in millions (not raw dollars)
   - User enters $150,000,000 → Store as 150

3. **IMDb IDs**: Directors and actors use IMDb IDs (e.g., nm0000116)
   - Use `name_mappings.json` to display real names in UI
   - Store IMDb IDs in backend

4. **Binary Features**: All genre/director/actor/country/company features are 0 or 1

5. **Validation**: Recommended input ranges:
   - Budget: $1M - $500M (1 - 500)
   - Runtime: 60 - 240 minutes
   - Year: 1980 - 2030

## 📈 Model Limitations

- **26% variance explained**: Profit is influenced by many factors beyond these features
- **±$262M average error**: Predictions are estimates, not guarantees
- **Best for mainstream movies**: Trained on movies with complete budget/revenue data
- **Historical data**: Based on past movies; may not predict future trends perfectly

## 🛠️ Troubleshooting

### Error: "Feature mismatch"
- Ensure all 190 features are present in input DataFrame
- Check feature order matches metadata file

### Error: "Model file not found"
- Verify `.pkl` and `.json` files are in the same directory as your script

### Unexpected predictions
- Check that budget is in millions (not raw dollars)
- Verify IMDb IDs are correct (use name_mappings.json)
- Ensure binary features are 0 or 1 (not True/False)

## 📝 Version History

- **v1.0** (2025-10-16): Initial deployment version
  - Linear Regression model
  - 190 features
  - Test R² = 0.26

## 📧 Support

For detailed implementation help, see `DEPLOYMENT_GUIDE.md`

For working code example, see `example_usage.py`

## 📄 License

Please provide appropriate attribution when using this model.

---

**Happy predicting! 🎬💰**
