# Movie Rating Prediction Model

This project builds a machine learning model to predict movie ratings based on various features including genre, directors, actors, and other metadata.

## Project Structure

```
movie-rating-predictor/
├── data/
│   ├── raw/                 # Raw datasets
│   └── processed/           # Cleaned and preprocessed data
├── notebooks/               # Jupyter notebooks for exploration
├── src/                     # Source code
│   ├── data_collection.py   # Data collection scripts
│   ├── preprocessing.py     # Data preprocessing
│   ├── feature_engineering.py # Feature engineering
│   ├── models.py           # Model definitions
│   └── evaluation.py       # Model evaluation
├── models/                  # Trained model artifacts
├── config/                  # Configuration files
└── requirements.txt         # Python dependencies
```

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Collection**:
   ```bash
   python src/data_collection.py
   ```

3. **Run Jupyter Notebooks**:
   ```bash
   jupyter notebook notebooks/
   ```

## Data Sources

- IMDb Datasets
- MovieLens
- The Movie Database (TMDb) API
- Rotten Tomatoes (if available)

## Model Features

- **Genre**: Movie genres (action, comedy, drama, etc.)
- **Directors**: Director reputation and historical performance
- **Actors**: Cast star power and popularity metrics
- **Temporal**: Release year, season effects
- **Production**: Budget, runtime, production company
- **Aggregate**: Historical ratings of cast/crew previous works

## Model Approaches

1. **Baseline Models**: Linear regression, mean prediction
2. **Tree-based Models**: Random Forest, Gradient Boosting
3. **Deep Learning**: Neural networks for complex interactions
4. **Ensemble Methods**: Combining multiple models

## Evaluation Metrics

- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)
- **Custom metrics** for different rating ranges

## Usage

See the notebooks in the `notebooks/` directory for detailed examples and analysis.