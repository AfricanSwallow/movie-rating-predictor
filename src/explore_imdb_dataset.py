"""
Simple script to explore the IMDB Reviews dataset structure from Hugging Face.
"""

import pandas as pd
from huggingface_hub import hf_hub_download
import os
import json

def explore_dataset():
    """Explore the dataset structure and download files manually."""
    
    # First, let's try to download and examine the dataset files manually
    dataset_name = "Daksh0505/IMDB-Reviews"
    
    try:
        # Let's try to get the dataset info
        from huggingface_hub import DatasetInfo, HfApi
        
        api = HfApi()
        dataset_info = api.dataset_info(dataset_name)
        
        print(f"📊 Dataset Info:")
        print(f"  ID: {dataset_info.id}")
        print(f"  Tags: {dataset_info.tags}")
        print(f"  Card Data: {dataset_info.cardData}")
        
        # List the files in the dataset
        files = api.list_repo_files(dataset_name, repo_type="dataset")
        print(f"\n📁 Dataset Files:")
        for file in files:
            print(f"  - {file}")
        
        # Try to download the dataset configuration file
        try:
            config_file = hf_hub_download(
                repo_id=dataset_name,
                filename="dataset_infos.json",
                repo_type="dataset"
            )
            
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            print(f"\n📋 Dataset Configuration:")
            print(json.dumps(config_data, indent=2))
            
        except Exception as e:
            print(f"Could not download config file: {e}")
        
        # Try to manually load with different approaches
        print(f"\n🔄 Trying alternative loading methods...")
        
        # Method 1: Try loading without split specification
        try:
            from datasets import load_dataset
            dataset = load_dataset(dataset_name, split="train")
            print(f"✅ Successfully loaded with split='train'")
            print(f"📊 Dataset: {dataset}")
            
            # Convert to pandas
            df = dataset.to_pandas()
            print(f"📊 DataFrame shape: {df.shape}")
            print(f"📋 Columns: {list(df.columns)}")
            
            # Show first few rows
            print(f"\n🔍 First 3 rows:")
            print(df.head(3))
            
            return df
            
        except Exception as e:
            print(f"❌ Method 1 failed: {e}")
        
        # Method 2: Try different format specification
        try:
            dataset = load_dataset(dataset_name, data_files="train.json")
            print(f"✅ Successfully loaded with data_files='train.json'")
            return dataset.to_pandas()
        except Exception as e:
            print(f"❌ Method 2 failed: {e}")
        
        # Method 3: Try to manually download a sample file
        try:
            # Look for JSON files in the repository
            json_files = [f for f in files if f.endswith('.json')]
            if json_files:
                sample_file = json_files[0]
                print(f"\n📥 Downloading sample file: {sample_file}")
                
                downloaded_file = hf_hub_download(
                    repo_id=dataset_name,
                    filename=sample_file,
                    repo_type="dataset"
                )
                
                with open(downloaded_file, 'r') as f:
                    sample_data = json.load(f)
                
                print(f"📋 Sample file structure:")
                print(json.dumps(sample_data, indent=2)[:1000] + "..." if len(str(sample_data)) > 1000 else json.dumps(sample_data, indent=2))
                
                return sample_data
                
        except Exception as e:
            print(f"❌ Method 3 failed: {e}")
            
    except Exception as e:
        print(f"❌ Failed to explore dataset: {e}")
        return None

if __name__ == "__main__":
    result = explore_dataset()
    if result is not None:
        print(f"\n🎉 Successfully explored the dataset!")
    else:
        print(f"\n❌ Failed to explore the dataset")