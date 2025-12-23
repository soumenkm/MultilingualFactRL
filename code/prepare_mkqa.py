"""
Helper script to download and prepare MKQA dataset for evaluation.
Converts MKQA format to the expected JSONL format.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
import requests
from tqdm import tqdm


class MKQADatasetPreparation:
    """
    Class to download and prepare MKQA dataset.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize dataset preparation.
        
        Args:
            config: Configuration dictionary containing:
                - output_dir: Directory to save processed data
                - download_url: URL to download MKQA data (optional)
                - raw_data_path: Path to raw MKQA JSON file (if already downloaded)
                - split_ratios: Dictionary with train/val/test ratios
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.output_dir = config.get("output_dir", "./mkqa_data")
        self.raw_data_path = config.get("raw_data_path", None)
        self.split_ratios = config.get("split_ratios", {
            "train": 0.8,
            "validation": 0.1,
            "test": 0.1
        })
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized MKQA dataset preparation")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def download_dataset(self) -> None:
        """
        Download MKQA dataset from GitHub.
        """
        self.logger.info("Downloading MKQA dataset...")
        
        # MKQA is available on GitHub
        base_url = "https://github.com/apple/ml-mkqa/raw/main/dataset"
        
        # Download main data file
        data_url = f"{base_url}/mkqa.jsonl.gz"
        output_dir = Path(self.output_dir)
        output_path = output_dir / "mkqa.jsonl.gz"
        
        self.logger.info(f"Downloading from: {data_url}")
        
        try:
            response = requests.get(data_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            self.logger.info(f"Downloaded to: {output_path}")
            
            # Decompress
            import gzip
            import shutil
            
            decompressed_path = output_dir / "mkqa.jsonl"
            self.logger.info("Decompressing...")
            
            with gzip.open(output_path, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            self.logger.info(f"Decompressed to: {decompressed_path}")
            self.raw_data_path = str(decompressed_path)
            
        except Exception as e:
            self.logger.error(f"Failed to download dataset: {e}")
            self.logger.info("\nAlternative: Download manually from:")
            self.logger.info("https://github.com/apple/ml-mkqa")
            raise
    
    def load_raw_data(self) -> List[Dict]:
        """
        Load raw MKQA data.
        
        Returns:
            List of MKQA samples
        """
        if not self.raw_data_path:
            self.logger.error("Raw data path not provided")
            raise FileNotFoundError(
                "MKQA data path not provided\n"
                "Please download from: https://github.com/apple/ml-mkqa"
            )
        
        data_path = Path(self.raw_data_path)
        
        if not data_path.exists():
            self.logger.error(f"Raw data not found at: {data_path}")
            raise FileNotFoundError(
                f"MKQA data not found at: {data_path}\n"
                "Please download from: https://github.com/apple/ml-mkqa"
            )
        
        self.logger.info(f"Loading raw data from: {data_path}")
        
        samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading samples"):
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        
        self.logger.info(f"Loaded {len(samples)} samples")
        return samples
    
    def filter_samples(self, samples: List[Dict]) -> List[Dict]:
        """
        Filter out samples with no answers or long answers only.
        
        Args:
            samples: List of MKQA samples
            
        Returns:
            Filtered list of samples
        """
        self.logger.info("Filtering samples...")
        
        filtered = []
        
        for sample in tqdm(samples, desc="Filtering"):
            # Check if sample has answers in English
            if "answers" not in sample or "en" not in sample["answers"]:
                continue
            
            en_answer = sample["answers"]["en"][0] if sample["answers"]["en"] else None
            
            # Skip if no answer or long answer type
            if not en_answer or en_answer.get("type") == "long_answer":
                continue
            
            # Keep samples with extractable answers
            if en_answer.get("text"):
                filtered.append(sample)
        
        self.logger.info(f"Filtered: {len(filtered)} samples (from {len(samples)})")
        return filtered
    
    def create_splits(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Create train/validation/test splits.
        
        Args:
            samples: List of MKQA samples
            
        Returns:
            Dictionary with split names as keys and sample lists as values
        """
        self.logger.info("Creating data splits...")
        
        import random
        random.seed(42)
        
        # Shuffle samples
        shuffled = samples.copy()
        random.shuffle(shuffled)
        
        # Calculate split sizes
        total = len(shuffled)
        train_size = int(total * self.split_ratios["train"])
        val_size = int(total * self.split_ratios["validation"])
        
        # Create splits
        splits = {
            "train": shuffled[:train_size],
            "validation": shuffled[train_size:train_size + val_size],
            "test": shuffled[train_size + val_size:],
        }
        
        for split_name, split_data in splits.items():
            self.logger.info(f"  {split_name}: {len(split_data)} samples")
        
        return splits
    
    def save_splits(self, splits: Dict[str, List[Dict]]) -> None:
        """
        Save splits to JSONL files.
        
        Args:
            splits: Dictionary with split data
        """
        self.logger.info("Saving splits...")
        
        output_dir = Path(self.output_dir)
        
        for split_name, split_data in splits.items():
            output_path = output_dir / f"{split_name}.jsonl"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in split_data:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            self.logger.info(f"  Saved {split_name}: {output_path}")
    
    def create_sample_file(self, samples: List[Dict], num_samples: int = 10) -> None:
        """
        Create a small sample file for testing.
        
        Args:
            samples: List of samples
            num_samples: Number of samples to include
        """
        output_dir = Path(self.output_dir)
        sample_path = output_dir / "sample.jsonl"
        
        with open(sample_path, 'w', encoding='utf-8') as f:
            for sample in samples[:num_samples]:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Created sample file with {num_samples} samples: {sample_path}")
    
    def prepare(self) -> None:
        """
        Run full dataset preparation pipeline.
        """
        self.logger.info("="*80)
        self.logger.info("Starting MKQA Dataset Preparation")
        self.logger.info("="*80)
        
        # Download if needed
        if not self.raw_data_path:
            self.download_dataset()
        else:
            data_path = Path(self.raw_data_path)
            if not data_path.exists():
                self.download_dataset()
        
        # Load raw data
        samples = self.load_raw_data()
        
        # Filter samples
        filtered_samples = self.filter_samples(samples)
        
        # Create splits
        splits = self.create_splits(filtered_samples)
        
        # Save splits
        self.save_splits(splits)
        
        # Create sample file
        self.create_sample_file(filtered_samples, num_samples=10)
        
        self.logger.info("="*80)
        self.logger.info("Dataset preparation complete!")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("="*80)


def main():
    """Main function to prepare MKQA dataset."""
    # Setup logging
    log_file = Path("./log") / "mkqa_preparation.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Configuration
    config = {
        "output_dir": "./data",
        "raw_data_path": "./data/mkqa.jsonl",  # Update if already have the file
        "split_ratios": {
            "train": 0.8,
            "validation": 0.1,
            "test": 0.1,
        }
    }
    
    logger.info("="*80)
    logger.info("MKQA Dataset Preparation")
    logger.info("="*80)
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("="*80)
    
    try:
        # Initialize and run preparation
        preparer = MKQADatasetPreparation(config)
        preparer.prepare()
        
        logger.info("\nDataset preparation completed successfully!")
        logger.info("\nYou can now run compute_metrics.py with:")
        logger.info("  config['dataset_path'] = './data/mkqa.jsonl'")
        
    except Exception as e:
        logger.error(f"Error during preparation: {e}", exc_info=True)
        logger.info("\nManual download instructions:")
        logger.info("1. Visit: https://github.com/apple/ml-mkqa")
        logger.info("2. Download mkqa.jsonl.gz")
        logger.info("3. Extract to ./data/mkqa.jsonl")
        logger.info("4. Run this script again with:")
        logger.info("   config['raw_data_path'] = './data/mkqa.jsonl'")
        raise


if __name__ == "__main__":
    main()