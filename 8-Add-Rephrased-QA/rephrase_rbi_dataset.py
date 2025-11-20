"""
RBI QA Dataset Rephraser - EXACT SCHEMA VERSION
Maintains original schema + adds only 'data_source' column
"""

import os
import json
import asyncio
from typing import List, Optional
from pathlib import Path
from datetime import datetime

import pandas as pd
from datasets import load_dataset
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from tqdm.asyncio import tqdm as async_tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Hugging Face dataset
    HF_DATASET = "Vishva007/RBI-Circular-QA-Dataset"
    TRAIN_SPLIT = "train"
    
    # Gemini API
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
    GEMINI_MODEL = "gemini-2.0-flash"
    
    # Processing
    NUM_REPHRASINGS = 3
    BATCH_SIZE = 50
    MAX_RETRIES = 3
    
    # Output
    OUTPUT_FILE = "Data/Rephrased/rbi_qa_rephrased.json"
    CHECKPOINT_FILE = "Data/Rephrased/checkpoint.json"


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class RephrasedQA(BaseModel):
    """Single rephrased QA pair"""
    rephrased_question: str = Field(
        description="A semantically equivalent but syntactically different version of the original question"
    )
    rephrased_answer: str = Field(
        description="A semantically equivalent but syntactically different version of the original answer"
    )


class RephrasedQAList(BaseModel):
    """List of rephrased QA pairs"""
    rephrasings: List[RephrasedQA] = Field(
        description=f"List of {Config.NUM_REPHRASINGS} rephrased question-answer pairs"
    )


# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

class CheckpointManager:
    """Manages progress tracking"""
    
    def __init__(self, checkpoint_file: str, output_file: str):
        self.checkpoint_file = checkpoint_file
        self.output_file = output_file
    
    def load_checkpoint(self) -> dict:
        """Load checkpoint state"""
        if Path(self.checkpoint_file).exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {
            "last_processed_index": -1,
            "total_processed": 0,
            "timestamp": None
        }
    
    def save_checkpoint(self, index: int, total: int):
        """Save checkpoint"""
        checkpoint = {
            "last_processed_index": index,
            "total_processed": total,
            "timestamp": datetime.now().isoformat()
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def load_existing_data(self) -> List[dict]:
        """Load existing data"""
        if Path(self.output_file).exists():
            with open(self.output_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_data(self, data: List[dict]):
        """Save data"""
        with open(self.output_file, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================================
# REPHRASER CLASS
# ============================================================================

class QARephraser:
    """Handles QA rephrasing"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.GEMINI_API_KEY,
            temperature=0.7,
            max_output_tokens=2048,
        )
        
        self.llm_with_structure = self.llm.with_structured_output(
            RephrasedQAList
        )
    
    def create_rephrasing_prompt(self, question: str, answer: str) -> str:
        """Create rephrasing prompt"""
        return f"""You are an expert at paraphrasing financial regulatory text while maintaining semantic equivalence.

Given the following question-answer pair about RBI (Reserve Bank of India) banking regulations, generate {Config.NUM_REPHRASINGS} diverse rephrasings.

IMPORTANT REQUIREMENTS:
1. Maintain EXACT semantic meaning - don't add or remove information
2. Use diverse syntactic structures (formal/casual, short/long, direct/indirect)
3. Vary vocabulary while keeping technical terms accurate
4. Vary question types (What/How/Explain/Describe/Why)
5. For answers: vary between detailed/concise, paragraphs/bullets, different ordering

ORIGINAL QUESTION:
{question}

ORIGINAL ANSWER:
{answer}

Generate {Config.NUM_REPHRASINGS} high-quality rephrasings that are semantically identical but syntactically diverse."""
    
    async def rephrase_single(self, question: str, answer: str, retries: int = 0) -> Optional[List[RephrasedQA]]:
        """Rephrase a single QA pair"""
        try:
            prompt = self.create_rephrasing_prompt(question, answer)
            result = await self.llm_with_structure.ainvoke(prompt)
            
            if len(result.rephrasings) != Config.NUM_REPHRASINGS:
                print(f"Warning: Expected {Config.NUM_REPHRASINGS}, got {len(result.rephrasings)}")
            
            return result.rephrasings
            
        except Exception as e:
            if retries < Config.MAX_RETRIES:
                print(f"Error (attempt {retries + 1}/{Config.MAX_RETRIES}): {str(e)}")
                await asyncio.sleep(2 ** retries)
                return await self.rephrase_single(question, answer, retries + 1)
            else:
                print(f"Failed after {Config.MAX_RETRIES} retries: {str(e)}")
                return None
    
    async def rephrase_batch(self, batch: List[dict]) -> List[dict]:
        """Rephrase a batch - MAINTAINS EXACT SCHEMA"""
        tasks = [
            self.rephrase_single(item['question'], item['answer'])
            for item in batch
        ]
        results = await asyncio.gather(*tasks)
        
        # CRITICAL: Maintain exact schema - only change question/answer fields
        rephrased_data = []
        for original, rephrasings in zip(batch, results):
            if rephrasings:
                for rephrased in rephrasings:
                    # Create new record with EXACT same schema
                    new_record = {
                        'document': original['document'],
                        'filename': original['filename'],
                        'model_name': original['model_name'],
                        'regulation_area': original['regulation_area'],
                        'applicable_to': original['applicable_to'],
                        'issued_on': original['issued_on'],
                        'key_topics': original['key_topics'],
                        'chunks_text': original['chunks_text'],
                        'is_table': original['is_table'],
                        'question': rephrased.rephrased_question,  # ONLY CHANGE
                        'answer': rephrased.rephrased_answer,      # ONLY CHANGE
                        'evaluation_criteria': original['evaluation_criteria'],
                        'category': original['category'],
                        'estimated_difficulty': original['estimated_difficulty'],
                        'rephrased_question': "",  # Keep empty like train split
                        'rephrased_answer': "",    # Keep empty like train split
                        'data_source': 'rephrased'  # NEW: only new column
                    }
                    rephrased_data.append(new_record)
        
        return rephrased_data


# ============================================================================
# MAIN PROCESSOR
# ============================================================================

class DatasetProcessor:
    """Main processor"""
    
    def __init__(self):
        self.rephraser = QARephraser()
        self.checkpoint_manager = CheckpointManager(
            Config.CHECKPOINT_FILE,
            Config.OUTPUT_FILE
        )
    
    def load_dataset(self) -> List[dict]:
        """Load dataset from Hugging Face"""
        print(f"Loading dataset from {Config.HF_DATASET}...")
        dataset = load_dataset(Config.HF_DATASET, split=Config.TRAIN_SPLIT)
        
        # Convert to list and add data_source column
        data = []
        for item in dataset:
            record = dict(item)
            # Ensure rephrased fields are empty strings (not NaN)
            if pd.isna(record.get('rephrased_question')):
                record['rephrased_question'] = ""
            if pd.isna(record.get('rephrased_answer')):
                record['rephrased_answer'] = ""
            record['data_source'] = 'original'  # Mark as original
            data.append(record)
        
        print(f"Loaded {len(data)} records from train split")
        return data
    
    async def process_dataset(self):
        """Main processing function"""
        # Load dataset
        original_data = self.load_dataset()
        
        # Load checkpoint and existing data
        checkpoint = self.checkpoint_manager.load_checkpoint()
        existing_data = self.checkpoint_manager.load_existing_data()
        
        start_index = checkpoint['last_processed_index'] + 1
        
        if start_index > 0:
            print(f"\n✓ Resuming from index {start_index}")
            print(f"✓ {checkpoint['total_processed']} records already processed")
            print(f"✓ Last checkpoint: {checkpoint['timestamp']}")
        else:
            print("\n✓ Starting fresh processing...")
            # Add all original records with data_source='original'
            existing_data = [record.copy() for record in original_data]
            self.checkpoint_manager.save_data(existing_data)
        
        # Process remaining data
        total_batches = (len(original_data) - start_index + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE
        print(f"\n✓ Processing {len(original_data) - start_index} records")
        print(f"✓ {total_batches} batches of {Config.BATCH_SIZE}")
        
        processed_count = checkpoint['total_processed']
        
        for batch_idx in range(0, len(original_data) - start_index, Config.BATCH_SIZE):
            current_idx = start_index + batch_idx
            batch = original_data[current_idx:current_idx + Config.BATCH_SIZE]
            
            print(f"\n{'='*70}")
            print(f"Batch {batch_idx // Config.BATCH_SIZE + 1}/{total_batches}")
            print(f"Records {current_idx} to {current_idx + len(batch) - 1}")
            print(f"{'='*70}")
            
            # Rephrase batch
            rephrased_batch = await self.rephraser.rephrase_batch(batch)
            
            # Add to dataset
            existing_data.extend(rephrased_batch)
            processed_count += len(batch)
            
            # Save progress
            self.checkpoint_manager.save_data(existing_data)
            self.checkpoint_manager.save_checkpoint(
                current_idx + len(batch) - 1,
                processed_count
            )
            
            print(f"✓ Generated {len(rephrased_batch)} rephrased records")
            print(f"✓ Total size: {len(existing_data)} records")
            print(f"✓ Progress: {processed_count}/{len(original_data)}")
            
            await asyncio.sleep(1)
        
        print(f"\n{'='*70}")
        print(f"✅ PROCESSING COMPLETE!")
        print(f"{'='*70}")
        
        # Final statistics
        df = pd.DataFrame(existing_data)
        print(f"\nOriginal records: {(df['data_source'] == 'original').sum()}")
        print(f"Rephrased records: {(df['data_source'] == 'rephrased').sum()}")
        print(f"Total dataset: {len(df)}")
        
        # Verify schema
        print(f"\n{'='*70}")
        print("SCHEMA VERIFICATION:")
        print(f"{'='*70}")
        expected_columns = [
            'document', 'filename', 'model_name', 'regulation_area',
            'applicable_to', 'issued_on', 'key_topics', 'chunks_text',
            'is_table', 'question', 'answer', 'evaluation_criteria',
            'category', 'estimated_difficulty', 'rephrased_question',
            'rephrased_answer', 'data_source'
        ]
        
        actual_columns = df.columns.tolist()
        if set(expected_columns) == set(actual_columns):
            print("✅ Schema is CORRECT!")
            print(f"✅ All {len(expected_columns)} columns present")
        else:
            print("❌ Schema mismatch!")
            print(f"Expected: {expected_columns}")
            print(f"Actual: {actual_columns}")
        
        print(f"\n✓ JSON saved: {Config.OUTPUT_FILE}")
        
        # Save CSV
        csv_file = Config.OUTPUT_FILE.replace('.json', '.csv')
        df.to_csv(csv_file, index=False)
        print(f"✓ CSV saved: {csv_file}")


# ============================================================================
# ENTRY POINT
# ============================================================================

async def main():
    """Main entry point"""
    print("="*70)
    print("RBI QA DATASET REPHRASER - EXACT SCHEMA VERSION")
    print("="*70)
    print(f"Model: {Config.GEMINI_MODEL}")
    print(f"Rephrasings per QA: {Config.NUM_REPHRASINGS}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Output: {Config.OUTPUT_FILE}")
    print("="*70)
    
    if Config.GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        print("\n❌ ERROR: Set GEMINI_API_KEY environment variable")
        print("   export GEMINI_API_KEY='your-api-key'")
        return
    
    processor = DatasetProcessor()
    await processor.process_dataset()


if __name__ == "__main__":
    asyncio.run(main())
