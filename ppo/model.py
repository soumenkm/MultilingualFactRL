import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Optional, Dict, Any
import os


class PPOModelManager:
    """
    Manages all models required for PPO training:
    - Policy model (trainable)
    - Reference model (frozen copy of initial policy)
    - Reward model (for computing rewards)
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B",
        use_4bit: bool = True,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        max_length: int = 512
    ):
        """
        Initialize PPO model manager.
        
        Args:
            model_name: HuggingFace model name for Llama 3.1
            use_4bit: Whether to use 4-bit quantization
            use_lora: Whether to use LoRA for efficient fine-tuning
            lora_r: LoRA attention dimension
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            device_map: Device map for model loading
            torch_dtype: Torch dtype for model
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_length = max_length
        
        self.tokenizer = None
        self.policy_model = None
        self.ref_model = None
        self.reward_model = None
        
        self._setup_quantization_config()
        self._load_tokenizer()
    
    def _setup_quantization_config(self) -> None:
        """Setup quantization configuration for 4-bit loading."""
        if self.use_4bit:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            print("4-bit quantization enabled")
        else:
            self.bnb_config = None
            print("Using full precision")
    
    def _load_tokenizer(self) -> None:
        """Load tokenizer for Llama 3.1."""
        print(f"Loading tokenizer from {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.tokenizer.padding_side = "left"  # Important for generation
        print("Tokenizer loaded successfully")
    
    def _setup_lora(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """
        Setup LoRA for the model.
        
        Args:
            model: Base model to apply LoRA to
            
        Returns:
            Model with LoRA applied
        """
        if not self.use_lora:
            return model
        
        print("Setting up LoRA...")
        
        # Prepare model for k-bit training if using quantization
        if self.use_4bit:
            model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def load_policy_model(self) -> None:
        """Load and setup the policy model (trainable)."""
        print(f"\nLoading policy model from {self.model_name}...")
        
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config if self.use_4bit else None,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        )
        
        # Apply LoRA if enabled
        self.policy_model = self._setup_lora(self.policy_model)
        
        print("Policy model loaded successfully")
    
    def load_reference_model(self) -> None:
        """Load reference model (frozen copy of initial policy)."""
        print(f"\nLoading reference model from {self.model_name}...")
        
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config if self.use_4bit else None,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        )
        
        # Reference model should not be trained
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        print("Reference model loaded successfully")
    
    def create_simple_reward_model(self) -> None:
        """
        Create a simple sentiment-based reward model.
        Uses a pre-trained sentiment classifier.
        """
        print("\nLoading reward model (sentiment classifier)...")
        
        # Use a sentiment analysis model as reward model
        reward_model_name = "lvwerra/distilbert-imdb"
        
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_name,
            num_labels=2
        )
        
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        
        # Move to device
        if torch.cuda.is_available():
            self.reward_model = self.reward_model.cuda()
        
        self.reward_model.eval()
        
        print("Reward model loaded successfully")
    
    def compute_reward(self, texts: list) -> torch.Tensor:
        """
        Compute rewards for generated texts using sentiment model.
        
        Args:
            texts: List of generated texts
            
        Returns:
            Tensor of reward scores
        """
        # Tokenize texts
        inputs = self.reward_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Get sentiment scores
        with torch.no_grad():
            outputs = self.reward_model(**inputs)
            # Get positive sentiment logit as reward
            rewards = outputs.logits[:, 1]  # Index 1 is positive sentiment
        
        return rewards
    
    def load_all_models(self) -> None:
        """Load all models required for PPO training."""
        self.load_policy_model()
        self.load_reference_model()
        self.create_simple_reward_model()
        print("\n" + "="*80)
        print("All models loaded successfully!")
        print("="*80)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models.
        
        Returns:
            Dictionary with model information
        """
        info = {
            "model_name": self.model_name,
            "use_4bit": self.use_4bit,
            "use_lora": self.use_lora,
            "max_length": self.max_length,
            "policy_model_loaded": self.policy_model is not None,
            "ref_model_loaded": self.ref_model is not None,
            "reward_model_loaded": self.reward_model is not None,
            "tokenizer_vocab_size": len(self.tokenizer) if self.tokenizer else None,
        }
        
        if self.use_lora and self.policy_model is not None:
            try:
                info["trainable_params"] = sum(
                    p.numel() for p in self.policy_model.parameters() if p.requires_grad
                )
                info["total_params"] = sum(
                    p.numel() for p in self.policy_model.parameters()
                )
            except:
                pass
        
        return info
    
    def save_policy_model(self, output_dir: str) -> None:
        """
        Save the trained policy model.
        
        Args:
            output_dir: Directory to save the model
        """
        print(f"\nSaving policy model to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        
        if self.use_lora:
            # Save LoRA adapters
            self.policy_model.save_pretrained(output_dir)
        else:
            # Save full model
            self.policy_model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        print("Model saved successfully")


class RewardModelWrapper:
    """
    Wrapper class for reward model to provide consistent interface.
    """
    
    def __init__(self, model_manager: PPOModelManager):
        """
        Initialize reward model wrapper.
        
        Args:
            model_manager: PPOModelManager instance
        """
        self.model_manager = model_manager
    
    def __call__(self, texts: list) -> torch.Tensor:
        """
        Compute rewards for texts.
        
        Args:
            texts: List of generated texts
            
        Returns:
            Tensor of rewards
        """
        return self.model_manager.compute_reward(texts)


def main():
    """Test model loading and setup."""
    print("=" * 80)
    print("Testing PPOModelManager")
    print("=" * 80)
    
    # Note: This will download large models, so it might take time
    # For testing, you might want to use a smaller model
    
    # Initialize model manager with smaller settings for testing
    model_manager = PPOModelManager(
        model_name="meta-llama/Meta-Llama-3.1-8B",  # You can change to smaller model for testing
        use_4bit=True,
        use_lora=True,
        lora_r=8,  # Smaller for testing
        max_length=256
    )
    
    print("\nAttempting to load all models...")
    print("(This may take several minutes for first-time download)")
    
    try:
        # Load all models
        model_manager.load_all_models()
        
        # Print model info
        print("\n" + "-" * 80)
        print("Model Information:")
        print("-" * 80)
        info = model_manager.get_model_info()
        for key, value in info.items():
            print(f"{key}: {value}")
        
        # Test tokenization
        print("\n" + "-" * 80)
        print("Testing tokenization:")
        print("-" * 80)
        sample_text = "This is a test prompt for the model."
        tokens = model_manager.tokenizer(
            sample_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        print(f"Input text: {sample_text}")
        print(f"Token IDs shape: {tokens['input_ids'].shape}")
        print(f"Decoded back: {model_manager.tokenizer.decode(tokens['input_ids'][0])}")
        
        # Test reward computation
        print("\n" + "-" * 80)
        print("Testing reward computation:")
        print("-" * 80)
        test_texts = [
            "This movie was absolutely amazing! I loved every moment.",
            "This was the worst film I have ever seen. Terrible."
        ]
        rewards = model_manager.compute_reward(test_texts)
        print(f"Test texts: {test_texts}")
        print(f"Rewards: {rewards.cpu().numpy()}")
        print("(Higher reward should be for positive text)")
        
        print("\n" + "=" * 80)
        print("Model testing complete!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during model loading: {e}")
        print("This is expected if you don't have access to Llama 3.1 or GPU")
        print("Make sure you:")
        print("1. Have accepted Llama 3.1 license on HuggingFace")
        print("2. Have logged in with HF token: huggingface-cli login")
        print("3. Have sufficient GPU memory or use smaller model")


if __name__ == "__main__":
    main()