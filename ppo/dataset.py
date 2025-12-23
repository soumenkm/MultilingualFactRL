import torch
from datasets import load_dataset
from typing import Dict, List, Optional
from torch.utils.data import Dataset


class IMDbPPODataset:
    """
    Class to handle IMDb dataset for PPO training.
    Creates prompts from movie reviews for sentiment-based reward learning.
    """
    
    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        prompt_template: str = "Review: {text}\nContinue this review in a positive tone:",
        min_length: int = 50,
        max_length: int = 200
    ):
        """
        Initialize IMDb dataset handler.
        
        Args:
            split: Dataset split to load ('train', 'test', 'unsupervised')
            max_samples: Maximum number of samples to load (None for all)
            prompt_template: Template for creating prompts from reviews
            min_length: Minimum length of review text to consider
            max_length: Maximum length of review text to consider
        """
        self.split = split
        self.max_samples = max_samples
        self.prompt_template = prompt_template
        self.min_length = min_length
        self.max_length = max_length
        
        self.dataset = None
        self.processed_prompts = []
        
        self._load_dataset()
        self._process_dataset()
    
    def _load_dataset(self) -> None:
        """Load IMDb dataset from HuggingFace."""
        print(f"Loading IMDb dataset ({self.split} split)...")
        self.dataset = load_dataset("imdb", split=self.split)
        
        if self.max_samples:
            self.dataset = self.dataset.select(range(min(self.max_samples, len(self.dataset))))
        
        print(f"Loaded {len(self.dataset)} samples")
    
    def _process_dataset(self) -> None:
        """Process dataset to create prompts for PPO training."""
        print("Processing dataset to create prompts...")
        
        for idx, example in enumerate(self.dataset):
            text = example["text"]
            
            # Filter by length
            if len(text) < self.min_length or len(text) > self.max_length:
                continue
            
            # Take first few sentences as prompt
            sentences = text.split('.')[:2]  # First 2 sentences
            prompt_text = '.'.join(sentences).strip()
            
            if len(prompt_text) > 20:  # Ensure prompt is meaningful
                prompt = self.prompt_template.format(text=prompt_text)
                self.processed_prompts.append({
                    "query": prompt,
                    "original_text": text,
                    "label": example["label"]
                })
        
        print(f"Created {len(self.processed_prompts)} prompts")
    
    def get_prompts(self) -> List[str]:
        """
        Get list of processed prompts for training.
        
        Returns:
            List of prompt strings
        """
        return [item["query"] for item in self.processed_prompts]
    
    def get_prompt_data(self) -> List[Dict]:
        """
        Get full prompt data including metadata.
        
        Returns:
            List of dictionaries with query, original_text, and label
        """
        return self.processed_prompts
    
    def __len__(self) -> int:
        """Return number of processed prompts."""
        return len(self.processed_prompts)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single prompt by index.
        
        Args:
            idx: Index of the prompt
            
        Returns:
            Dictionary with prompt data
        """
        return self.processed_prompts[idx]
    
    def get_batch(self, batch_size: int, shuffle: bool = True) -> List[str]:
        """
        Get a batch of prompts.
        
        Args:
            batch_size: Number of prompts to return
            shuffle: Whether to shuffle before selecting
            
        Returns:
            List of prompt strings
        """
        import random
        
        prompts = self.get_prompts()
        
        if shuffle:
            indices = random.sample(range(len(prompts)), min(batch_size, len(prompts)))
            return [prompts[i] for i in indices]
        else:
            return prompts[:batch_size]


class PPODatasetWrapper(Dataset):
    """
    PyTorch Dataset wrapper for IMDb PPO dataset.
    Useful for DataLoader integration.
    """
    
    def __init__(self, ppo_dataset: IMDbPPODataset):
        """
        Initialize wrapper.
        
        Args:
            ppo_dataset: IMDbPPODataset instance
        """
        self.ppo_dataset = ppo_dataset
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.ppo_dataset)
    
    def __getitem__(self, idx: int) -> str:
        """
        Get prompt by index.
        
        Args:
            idx: Index of the prompt
            
        Returns:
            Prompt string
        """
        return self.ppo_dataset[idx]["query"]


def main():
    """Test the dataset loading and processing."""
    print("=" * 80)
    print("Testing IMDbPPODataset")
    print("=" * 80)
    
    # Initialize dataset
    dataset = IMDbPPODataset(
        split="train",
        max_samples=100,  # Small sample for testing
        min_length=50,
        max_length=150
    )
    
    print(f"\nDataset length: {len(dataset)}")
    
    # Test getting single prompt
    print("\n" + "-" * 80)
    print("Sample prompt (index 0):")
    print("-" * 80)
    sample = dataset[0]
    print(f"Query: {sample['query'][:200]}...")
    print(f"Label: {sample['label']}")
    
    # Test getting batch
    print("\n" + "-" * 80)
    print("Testing batch retrieval (5 prompts):")
    print("-" * 80)
    batch = dataset.get_batch(batch_size=5, shuffle=True)
    for i, prompt in enumerate(batch, 1):
        print(f"\nPrompt {i}:")
        print(prompt[:150] + "...")
    
    # Test PyTorch wrapper
    print("\n" + "-" * 80)
    print("Testing PyTorch Dataset wrapper:")
    print("-" * 80)
    torch_dataset = PPODatasetWrapper(dataset)
    print(f"Wrapper length: {len(torch_dataset)}")
    print(f"Sample from wrapper: {torch_dataset[0][:100]}...")
    
    print("\n" + "=" * 80)
    print("Dataset testing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()