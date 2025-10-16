# Review Clustering Page - Integration Instructions

## 📦 What's Included

This branch contains a single-page Streamlit component for review clustering analysis:

**Main File:**
- `review_clustering_page.py` - Single-page Streamlit component

**Data Files:**
- `data/streamlit_export/positive_reviews_with_clusters.csv` (5,000 reviews with t-SNE coordinates)
- `data/streamlit_export/aspect_cluster_profiles.csv` (6 clusters × 10 aspects)
- `data/streamlit_export/genre_aspect_profiles.csv` (5 genres × 10 aspects)

**Example & Dependencies:**
- `integration_example.py` - Working code example showing how to integrate
- `streamlit_requirements.txt` - Required Python packages

---

## 🚀 How to Integrate

### Step 1: Install Dependencies

```bash
pip install -r streamlit_requirements.txt
```

Or install individually:
```bash
pip install streamlit pandas numpy matplotlib seaborn wordcloud scikit-learn
```

### Step 2: Import and Call

In your main Streamlit app file:

```python
from review_clustering_page import render_review_clustering_page

# In your page navigation logic
if page == "Review Clustering":
    render_review_clustering_page()  # That's it!
```

### Complete Example

See `integration_example.py` for a full working example of a multi-page app.

---

## 🎯 What the Page Shows

The page displays 4 main visualizations:

1. **☁️ Word Cloud Analysis** - Frequent terms in each review cluster (6 word clouds)
2. **🗺️ Cluster Visualization (t-SNE)** - 2D scatter plot of review clusters
3. **🎯 Aspect-Based Analysis** - Heatmap showing what viewers value most
4. **🎭 Genre Insights** - Genre-aspect preferences with interactive comparison

Users can switch between sections using radio buttons.

---

## 📁 File Structure

```
your_app/
├── main_app.py
├── review_clustering_page.py          # ← The page component
├── integration_example.py             # ← Reference example
├── requirements.txt
└── data/
    └── streamlit_export/              # ← Data files
        ├── positive_reviews_with_clusters.csv
        ├── aspect_cluster_profiles.csv
        └── genre_aspect_profiles.csv
```

---

## 🎨 Customization

Edit `review_clustering_page.py` to customize:

- **Colors**: Change the `colors` arrays in visualization functions
- **Layout**: Modify the CSS in `inject_custom_css()`
- **Text**: Update titles, descriptions, and insights
- **Data paths**: Change file paths if data is stored elsewhere

---

## 🔍 Troubleshooting

**"Data file not found" error:**
- Ensure `data/streamlit_export/` folder exists relative to where you run the app
- Check that all 3 CSV files are present

**Import errors:**
- Install missing packages: `pip install -r streamlit_requirements.txt`

**Page not displaying:**
- Make sure you're calling `render_review_clustering_page()` 
- Check terminal output for error messages

---

## 📊 Technical Details

- **Data processing**: Text preprocessing, tokenization, stemming
- **Clustering**: K-means on TF-IDF vectors (5 clusters)
- **Aspect analysis**: 10 key aspects (plot, acting, visuals, emotions, action, music, direction, originality, entertainment, pacing)
- **Dimensionality reduction**: t-SNE for 2D visualization

---

That's it! The page is self-contained and ready to integrate into your multi-page app. 🎬
