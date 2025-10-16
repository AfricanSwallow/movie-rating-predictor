# Instructions to Share Model Files with Git LFS

## Problem
The `.pkl` model files are in `.gitignore`, so they won't be pushed to GitHub.

## Solution 1: Git LFS (Large File Storage) - RECOMMENDED

Git LFS is designed for versioning large files like models.

### Setup (One-time)
```bash
# Install Git LFS (if not already installed)
# macOS:
brew install git-lfs

# Linux:
sudo apt-get install git-lfs

# Windows:
# Download from https://git-lfs.github.com/

# Initialize Git LFS in your repo
cd /Users/mexx/School/交換/Courses/DataScience/mini-project/movie-rating-predictor
git lfs install
```

### Track Model Files with LFS
```bash
# Remove .pkl from .gitignore (or create exception)
# Then track specific model files with LFS
git lfs track "models/profit_prediction_model.pkl"
git lfs track "models/profit_prediction_metadata.json"

# This creates/updates .gitattributes
git add .gitattributes
git add models/profit_prediction_model.pkl
git add models/profit_prediction_metadata.json
git add models/name_mappings.json
git commit -m "Add profit prediction model with Git LFS"
git push
```

### Your Friend Downloads
```bash
git clone <repo-url>
# Files are automatically downloaded via LFS
```

---

## Solution 2: Create Exception in .gitignore - SIMPLE

Add the profit prediction files as exceptions to `.gitignore`:

### Update .gitignore
Add these lines at the end of `.gitignore`:

```
# Exception: Include profit prediction model for deployment
!models/profit_prediction_model.pkl
!models/profit_prediction_metadata.json
!models/name_mappings.json
!models/DEPLOYMENT_GUIDE.md
!models/example_usage.py
!models/README.md
```

### Then commit
```bash
git add models/profit_prediction_model.pkl
git add models/profit_prediction_metadata.json
git add models/name_mappings.json
git add models/*.md
git add models/*.py
git commit -m "Add profit prediction model for deployment"
git push
```

---

## Solution 3: External File Sharing - FASTEST

Don't use Git at all for model files. Share via:

### 3a. Google Drive / Dropbox
```bash
# Create a ZIP file
cd models
zip -r profit_prediction_model.zip \
    profit_prediction_model.pkl \
    profit_prediction_metadata.json \
    name_mappings.json \
    DEPLOYMENT_GUIDE.md \
    example_usage.py \
    README.md

# Upload to Google Drive/Dropbox and share link
```

### 3b. GitHub Release
```bash
# Create a release on GitHub
# Upload the ZIP file as a release asset
# Your friend downloads from the Releases page
```

### 3c. Direct Transfer
```bash
# Email or Slack the ZIP file directly
```

---

## Solution 4: Model Registry (Production)

For production deployments, use a model registry:
- **MLflow**: Track and share models
- **DVC**: Data Version Control
- **AWS S3 / Google Cloud Storage**: Cloud storage
- **Hugging Face Hub**: Public model sharing

---

## Recommendation

**For your use case** (sharing with a friend for a web app):

1. **If file is small (<10MB)**: Use **Solution 2** (gitignore exception)
2. **If file is large (>10MB)**: Use **Solution 1** (Git LFS)
3. **If urgent/simple**: Use **Solution 3** (Zip + Drive/Dropbox)

## Current File Sizes

Let me check the actual sizes:

```bash
ls -lh models/profit_prediction_model.pkl
ls -lh models/profit_prediction_metadata.json
```

If < 10MB total → Solution 2 is easiest
If > 10MB total → Solution 1 (Git LFS) is better
