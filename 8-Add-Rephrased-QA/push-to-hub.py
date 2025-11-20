"""
Push Rephrased RBI Dataset to Hugging Face Hub - SCHEMA COMPATIBLE
Adds missing 'data_source' column to eval split for compatibility
"""

import os
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Input files (from rephrasing script)
    INPUT_JSON = "Data/Rephrased/rbi_qa_rephrased.json"
    INPUT_CSV = "Data/Rephrased/rbi_qa_rephrased.csv"
    
    # Hugging Face
    HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN_HERE")
    HF_REPO = "Vishva007/RBI-Circular-QA-Dataset"
    
    # Processing
    GROUP_BY_COLUMN = "filename"
    
    # Output
    OUTPUT_DIR = "Data/Grouped"


# ============================================================================
# DATASET GROUPER
# ============================================================================

class DatasetGrouper:
    """Groups and organizes dataset for upload"""
    
    def __init__(self):
        self.df = None
        self.grouped_data = defaultdict(list)
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load rephrased dataset"""
        print(f"Loading dataset from {filepath}...")
        
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.df = pd.DataFrame(data)
        elif filepath.endswith('.csv'):
            self.df = pd.read_csv(filepath)
        else:
            raise ValueError("File must be JSON or CSV")
        
        print(f"✓ Loaded {len(self.df)} records")
        return self.df
    
    def analyze_dataset(self):
        """Analyze dataset composition"""
        print(f"\n{'='*70}")
        print("DATASET ANALYSIS")
        print(f"{'='*70}")
        
        print(f"\nTotal records: {len(self.df)}")
        print(f"\nData source breakdown:")
        print(self.df['data_source'].value_counts())
        
        print(f"\nUnique {Config.GROUP_BY_COLUMN}: {self.df[Config.GROUP_BY_COLUMN].nunique()}")
        
        print(f"\nRecords per {Config.GROUP_BY_COLUMN}:")
        filename_counts = self.df[Config.GROUP_BY_COLUMN].value_counts()
        print(f"  Min: {filename_counts.min()}")
        print(f"  Max: {filename_counts.max()}")
        print(f"  Mean: {filename_counts.mean():.1f}")
        
        print(f"\nSchema columns: {len(self.df.columns)}")
        print(f"Columns: {self.df.columns.tolist()}")
    
    def group_by_filename(self):
        """Group records by filename"""
        print(f"\n{'='*70}")
        print(f"GROUPING BY {Config.GROUP_BY_COLUMN}")
        print(f"{'='*70}")
        
        grouped = self.df.groupby(Config.GROUP_BY_COLUMN)
        
        print(f"\nCreating {len(grouped)} groups...")
        
        for filename, group_df in grouped:
            records = group_df.to_dict('records')
            self.grouped_data[filename] = records
            
            if len(self.grouped_data) <= 3:
                print(f"\n  {filename}:")
                print(f"    Records: {len(records)}")
                print(f"    Original: {sum(1 for r in records if r['data_source'] == 'original')}")
                print(f"    Rephrased: {sum(1 for r in records if r['data_source'] == 'rephrased')}")
        
        print(f"\n✓ Created {len(self.grouped_data)} filename groups")
        return self.grouped_data
    
    def save_grouped_locally(self):
        """Save grouped data locally"""
        output_dir = Path(Config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print("SAVING GROUPED DATA LOCALLY")
        print(f"{'='*70}")
        
        for filename, records in self.grouped_data.items():
            safe_filename = filename.replace('/', '_').replace('\\', '_')
            output_path = output_dir / f"{safe_filename}.json"
            
            with open(output_path, 'w') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {len(self.grouped_data)} files to {Config.OUTPUT_DIR}")


# ============================================================================
# HUGGINGFACE UPLOADER
# ============================================================================

class HuggingFaceUploader:
    """Uploads dataset to Hugging Face Hub - SCHEMA COMPATIBLE"""
    
    def __init__(self, token: str):
        self.token = token
        self.logged_in = False
        self.existing_eval = None
    
    def login_to_hub(self):
        """Login to Hugging Face Hub"""
        print(f"\n{'='*70}")
        print("LOGGING IN TO HUGGING FACE HUB")
        print(f"{'='*70}")
        
        try:
            login(token=self.token)
            self.logged_in = True
            print("✓ Successfully logged in to Hugging Face Hub")
        except Exception as e:
            print(f"❌ Login failed: {str(e)}")
            raise
    
    def load_existing_eval_split(self, repo_id: str):
        """Load existing eval split and add missing columns"""
        print(f"\n{'='*70}")
        print("LOADING & UPDATING EVAL SPLIT")
        print(f"{'='*70}")
        
        try:
            print(f"Fetching eval split from {repo_id}...")
            existing_dataset = load_dataset(repo_id, split='eval')
            
            print(f"✓ Loaded existing eval split: {len(existing_dataset)} records")
            print(f"  Original columns: {list(existing_dataset.features.keys())}")
            
            # Convert to pandas to add missing column
            eval_df = existing_dataset.to_pandas()
            
            # Check if data_source column exists
            if 'data_source' not in eval_df.columns:
                print(f"\n⚠️  'data_source' column missing - adding it now...")
                eval_df['data_source'] = 'original'  # Mark all eval records as original
                print(f"✓ Added 'data_source' column (value: 'original')")
            
            # Convert back to Dataset
            self.existing_eval = Dataset.from_pandas(eval_df, preserve_index=False)
            
            print(f"✓ Updated eval split columns: {list(self.existing_eval.features.keys())}")
            return True
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load eval split: {str(e)}")
            print("   This is normal if your dataset doesn't have an eval split yet")
            self.existing_eval = None
            return False
    
    def create_dataset_dict(self, train_df: pd.DataFrame) -> DatasetDict:
        """Create DatasetDict with BOTH train and eval splits"""
        print(f"\n{'='*70}")
        print("CREATING HUGGING FACE DATASET")
        print(f"{'='*70}")
        
        # Convert train DataFrame to Dataset
        train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
        print(f"✓ Created new train split: {len(train_dataset)} records")
        print(f"  Train columns: {list(train_dataset.features.keys())}")
        
        # Verify schema compatibility
        if self.existing_eval is not None:
            train_cols = set(train_dataset.features.keys())
            eval_cols = set(self.existing_eval.features.keys())
            
            print(f"\n{'='*70}")
            print("SCHEMA COMPATIBILITY CHECK")
            print(f"{'='*70}")
            
            if train_cols == eval_cols:
                print("✅ Schema is COMPATIBLE!")
                print(f"   Both splits have {len(train_cols)} columns")
            else:
                print("⚠️  Schema mismatch detected:")
                print(f"   Train only: {train_cols - eval_cols}")
                print(f"   Eval only: {eval_cols - train_cols}")
                print("\n   Attempting to align schemas...")
                
                # Add missing columns to eval if needed
                eval_df = self.existing_eval.to_pandas()
                for col in train_cols - eval_cols:
                    print(f"   Adding '{col}' to eval split")
                    if col == 'data_source':
                        eval_df[col] = 'original'
                    else:
                        eval_df[col] = ""  # Empty string for other columns
                
                self.existing_eval = Dataset.from_pandas(eval_df, preserve_index=False)
                print("✅ Schemas aligned successfully!")
        
        # Create DatasetDict with BOTH splits
        if self.existing_eval is not None:
            dataset_dict = DatasetDict({
                "train": train_dataset,
                "eval": self.existing_eval
            })
            print(f"\n✓ Preserved eval split: {len(self.existing_eval)} records")
        else:
            dataset_dict = DatasetDict({
                "train": train_dataset
            })
            print("\n⚠️  No eval split to preserve (creating train-only dataset)")
        
        print(f"\nFinal dataset splits:")
        for split_name, split_data in dataset_dict.items():
            print(f"  {split_name}: {len(split_data)} records")
        
        return dataset_dict
    
    def push_to_hub(self, dataset_dict: DatasetDict, repo_id: str):
        """Push dataset to Hugging Face Hub"""
        print(f"\n{'='*70}")
        print("PUSHING TO HUGGING FACE HUB")
        print(f"{'='*70}")
        print(f"\nTarget repository: {repo_id}")
        print(f"This will UPDATE the repository with:")
        for split_name, split_data in dataset_dict.items():
            print(f"  - {split_name}: {len(split_data)} records")
        
        # Ask for confirmation
        response = input("\nDo you want to proceed? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Upload cancelled by user")
            return False
        
        try:
            print("\n⏳ Uploading dataset... (this may take several minutes)")
            
            dataset_dict.push_to_hub(
                repo_id=repo_id,
                token=self.token,
                private=False,
            )
            
            print(f"\n✅ Successfully pushed dataset to {repo_id}")
            print(f"🔗 View at: https://huggingface.co/datasets/{repo_id}")
            
            # Show final schema
            print(f"\n{'='*70}")
            print("FINAL DATASET SCHEMA")
            print(f"{'='*70}")
            for split_name, split_data in dataset_dict.items():
                print(f"\n{split_name} split:")
                print(f"  Records: {len(split_data)}")
                print(f"  Columns: {list(split_data.features.keys())}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Upload failed: {str(e)}")
            raise


# ============================================================================
# MAIN PROCESSOR
# ============================================================================

class MainProcessor:
    """Main processing pipeline"""
    
    def __init__(self):
        self.grouper = DatasetGrouper()
        self.uploader = None
    
    def run(self):
        """Execute full pipeline"""
        print("="*70)
        print("RBI QA DATASET - HUGGING FACE UPLOADER (SCHEMA SAFE)")
        print("="*70)
        
        # Step 1: Load data
        if Path(Config.INPUT_JSON).exists():
            df = self.grouper.load_data(Config.INPUT_JSON)
        elif Path(Config.INPUT_CSV).exists():
            df = self.grouper.load_data(Config.INPUT_CSV)
        else:
            print(f"❌ Error: No input file found")
            print(f"   Looked for: {Config.INPUT_JSON} or {Config.INPUT_CSV}")
            return
        
        # Step 2: Analyze dataset
        self.grouper.analyze_dataset()
        
        # Step 3: Group by filename (optional)
        grouped_data = self.grouper.group_by_filename()
        
        # Step 4: Save grouped data locally (optional)
        save_locally = input("\n💾 Save grouped data locally? (yes/no): ")
        if save_locally.lower() == 'yes':
            self.grouper.save_grouped_locally()
        
        # Step 5: Upload to Hugging Face
        print(f"\n{'='*70}")
        print("HUGGING FACE UPLOAD")
        print(f"{'='*70}")
        
        proceed = input("\nProceed with upload to Hugging Face? (yes/no): ")
        if proceed.lower() != 'yes':
            print("✓ Skipping upload")
            return
        
        # Check HF token
        if Config.HF_TOKEN == "YOUR_HF_TOKEN_HERE":
            print("\n❌ Error: HF_TOKEN not set")
            print("   Set environment variable: export HF_TOKEN='your-token'")
            return
        
        # Initialize uploader
        self.uploader = HuggingFaceUploader(Config.HF_TOKEN)
        self.uploader.login_to_hub()
        
        # Load existing eval split and add missing columns
        self.uploader.load_existing_eval_split(Config.HF_REPO)
        
        # Create dataset with compatible schemas
        dataset_dict = self.uploader.create_dataset_dict(df)
        
        # Upload
        self.uploader.push_to_hub(dataset_dict, Config.HF_REPO)
        
        print(f"\n{'='*70}")
        print("✅ PROCESSING COMPLETE")
        print(f"{'='*70}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    processor = MainProcessor()
    processor.run()


if __name__ == "__main__":
    main()
