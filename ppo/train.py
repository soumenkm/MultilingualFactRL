import torch
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import GenerationConfig
from typing import Optional, Dict, List, Any
import wandb
from datetime import datetime
import os
import json

from dataset import IMDbPPODataset
from model import PPOModelManager, RewardModelWrapper


class PPOTrainingConfig:
    """
    Configuration class for PPO training.
    Wraps PPOConfig and adds additional settings.
    """
    
    def __init__(
        self,
        # Model settings
        model_name: str = "meta-llama/Meta-Llama-3.1-8B",
        use_4bit: bool = True,
        use_lora: bool = True,
        
        # Training settings
        learning_rate: float = 1.41e-5,
        batch_size: int = 4,
        mini_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        num_train_epochs: int = 1,
        max_steps: int = 1000,
        
        # PPO specific
        ppo_epochs: int = 4,
        init_kl_coef: float = 0.2,
        target_kl: float = 6.0,
        adap_kl_ctrl: bool = True,
        gamma: float = 1.0,
        lam: float = 0.95,
        cliprange: float = 0.2,
        cliprange_value: float = 0.2,
        vf_coef: float = 0.1,
        
        # Generation settings
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        
        # Logging
        log_with: Optional[str] = None,  # "wandb" or None
        logging_steps: int = 10,
        save_steps: int = 100,
        eval_steps: int = 50,
        
        # Output
        output_dir: str = "./ppo_llama_outputs",
        run_name: Optional[str] = None,
        
        # Dataset
        max_dataset_samples: int = 5000,
    ):
        """Initialize training configuration."""
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.use_lora = use_lora
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        
        self.ppo_epochs = ppo_epochs
        self.init_kl_coef = init_kl_coef
        self.target_kl = target_kl
        self.adap_kl_ctrl = adap_kl_ctrl
        self.gamma = gamma
        self.lam = lam
        self.cliprange = cliprange
        self.cliprange_value = cliprange_value
        self.vf_coef = vf_coef
        
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        
        self.log_with = log_with
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        
        self.output_dir = output_dir
        self.run_name = run_name or f"ppo_llama_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.max_dataset_samples = max_dataset_samples
    
    def to_ppo_config(self) -> PPOConfig:
        """
        Convert to TRL PPOConfig object.
        
        Returns:
            PPOConfig instance
        """
        return PPOConfig(
            model_name=self.model_name,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            mini_batch_size=self.mini_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            ppo_epochs=self.ppo_epochs,
            init_kl_coef=self.init_kl_coef,
            target=self.target_kl,
            adap_kl_ctrl=self.adap_kl_ctrl,
            gamma=self.gamma,
            lam=self.lam,
            cliprange=self.cliprange,
            cliprange_value=self.cliprange_value,
            vf_coef=self.vf_coef,
            log_with=self.log_with,
            seed=42,
            optimize_cuda_cache=True,
        )
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def get_generation_config(self) -> GenerationConfig:
        """
        Get generation configuration.
        
        Returns:
            GenerationConfig instance
        """
        return GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            do_sample=self.do_sample,
            pad_token_id=None,  # Will be set from tokenizer
        )


class LlamaPPOTrainer:
    """
    Main PPO trainer class for Llama 3.1.
    Orchestrates the entire training process.
    """
    
    def __init__(self, config: PPOTrainingConfig):
        """
        Initialize PPO trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.model_manager = None
        self.dataset = None
        self.ppo_trainer = None
        self.generation_config = None
        
        self._setup_output_dir()
        self._initialize_wandb()
    
    def _setup_output_dir(self) -> None:
        """Create output directory structure."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Save config
        config_path = os.path.join(self.config.output_dir, "training_config.json")
        self.config.save(config_path)
        print(f"Saved training configuration to {config_path}")
    
    def _initialize_wandb(self) -> None:
        """Initialize Weights & Biases logging if enabled."""
        if self.config.log_with == "wandb":
            wandb.init(
                project="llama-ppo-training",
                name=self.config.run_name,
                config=self.config.__dict__
            )
            print("Initialized W&B logging")
    
    def setup_models(self) -> None:
        """Setup all required models."""
        print("\n" + "="*80)
        print("Setting up models...")
        print("="*80)
        
        self.model_manager = PPOModelManager(
            model_name=self.config.model_name,
            use_4bit=self.config.use_4bit,
            use_lora=self.config.use_lora,
        )
        
        # Load policy model (will be wrapped with value head)
        self.model_manager.load_policy_model()
        
        # Load reference model
        self.model_manager.load_reference_model()
        
        # Load reward model
        self.model_manager.create_simple_reward_model()
        
        print("Models setup complete!")
    
    def setup_dataset(self) -> None:
        """Setup training dataset."""
        print("\n" + "="*80)
        print("Setting up dataset...")
        print("="*80)
        
        self.dataset = IMDbPPODataset(
            split="train",
            max_samples=self.config.max_dataset_samples,
        )
        
        print(f"Dataset ready with {len(self.dataset)} prompts")
    
    def setup_trainer(self) -> None:
        """Setup PPO trainer from TRL."""
        print("\n" + "="*80)
        print("Setting up PPO trainer...")
        print("="*80)
        
        # Convert policy model to CausalLM with value head
        policy_model_with_value = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.model_manager.policy_model
        )
        
        # Get PPO config
        ppo_config = self.config.to_ppo_config()
        
        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=policy_model_with_value,
            ref_model=self.model_manager.ref_model,
            tokenizer=self.model_manager.tokenizer,
        )
        
        # Setup generation config
        self.generation_config = self.config.get_generation_config()
        self.generation_config.pad_token_id = self.model_manager.tokenizer.pad_token_id
        
        print("PPO trainer initialized!")
    
    def train(self) -> None:
        """Execute PPO training loop."""
        print("\n" + "="*80)
        print("Starting PPO Training")
        print("="*80)
        
        prompts = self.dataset.get_prompts()
        total_steps = 0
        
        for epoch in range(self.config.num_train_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{self.config.num_train_epochs}")
            print(f"{'='*80}")
            
            # Iterate through batches
            for step in range(0, len(prompts), self.config.batch_size):
                if total_steps >= self.config.max_steps:
                    print(f"\nReached max steps ({self.config.max_steps}). Stopping training.")
                    return
                
                # Get batch of prompts
                batch_prompts = prompts[step:step + self.config.batch_size]
                
                # Tokenize prompts
                query_tensors = [
                    self.model_manager.tokenizer.encode(prompt, return_tensors="pt")[0]
                    for prompt in batch_prompts
                ]
                
                # Generate responses
                response_tensors = self.ppo_trainer.generate(
                    query_tensors,
                    return_prompt=False,
                    generation_config=self.generation_config,
                )
                
                # Decode responses
                batch_responses = [
                    self.model_manager.tokenizer.decode(r.squeeze(), skip_special_tokens=True)
                    for r in response_tensors
                ]
                
                # Compute rewards using reward model
                rewards = self.model_manager.compute_reward(batch_responses)
                rewards = [r for r in rewards]  # Convert to list
                
                # Run PPO step
                stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
                
                total_steps += 1
                
                # Logging
                if total_steps % self.config.logging_steps == 0:
                    self._log_training_stats(stats, batch_prompts, batch_responses, rewards, total_steps)
                
                # Save checkpoint
                if total_steps % self.config.save_steps == 0:
                    self._save_checkpoint(total_steps)
        
        print("\n" + "="*80)
        print("Training completed!")
        print("="*80)
        
        # Save final model
        self._save_final_model()
    
    def _log_training_stats(
        self,
        stats: Dict[str, Any],
        prompts: List[str],
        responses: List[str],
        rewards: List[float],
        step: int
    ) -> None:
        """
        Log training statistics.
        
        Args:
            stats: Training statistics from PPO step
            prompts: Batch prompts
            responses: Generated responses
            rewards: Computed rewards
            step: Current training step
        """
        print(f"\n{'='*80}")
        print(f"Step {step}")
        print(f"{'='*80}")
        
        # Print key metrics
        if 'ppo/loss/total' in stats:
            print(f"Total Loss: {stats['ppo/loss/total']:.4f}")
        if 'ppo/policy/approxkl' in stats:
            print(f"Approx KL: {stats['ppo/policy/approxkl']:.4f}")
        if 'ppo/returns/mean' in stats:
            print(f"Mean Return: {stats['ppo/returns/mean']:.4f}")
        
        print(f"Mean Reward: {sum(rewards) / len(rewards):.4f}")
        
        # Print sample generation
        if len(prompts) > 0:
            print(f"\n{'-'*80}")
            print("Sample Generation:")
            print(f"{'-'*80}")
            print(f"Prompt: {prompts[0][:100]}...")
            print(f"Response: {responses[0][:200]}...")
            print(f"Reward: {rewards[0]:.4f}")
        
        # Log to wandb if enabled
        if self.config.log_with == "wandb":
            wandb.log({
                "step": step,
                "mean_reward": sum(rewards) / len(rewards),
                **stats
            })
    
    def _save_checkpoint(self, step: int) -> None:
        """
        Save training checkpoint.
        
        Args:
            step: Current training step
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{step}")
        print(f"\nSaving checkpoint to {checkpoint_path}...")
        self.ppo_trainer.save_pretrained(checkpoint_path)
        print("Checkpoint saved!")
    
    def _save_final_model(self) -> None:
        """Save final trained model."""
        final_model_path = os.path.join(self.config.output_dir, "final_model")
        print(f"\nSaving final model to {final_model_path}...")
        self.ppo_trainer.save_pretrained(final_model_path)
        self.model_manager.tokenizer.save_pretrained(final_model_path)
        print("Final model saved!")
    
    def run(self) -> None:
        """
        Run the complete training pipeline.
        """
        print("\n" + "="*80)
        print("Starting Llama 3.1 PPO Training Pipeline")
        print("="*80)
        
        # Setup
        self.setup_models()
        self.setup_dataset()
        self.setup_trainer()
        
        # Train
        self.train()
        
        # Cleanup
        if self.config.log_with == "wandb":
            wandb.finish()
        
        print("\n" + "="*80)
        print("Pipeline completed successfully!")
        print(f"Output directory: {self.config.output_dir}")
        print("="*80)


def main():
    """
    Main function to run PPO training.
    """
    print("="*80)
    print("Llama 3.1 PPO Training with TRL")
    print("="*80)
    
    # Create training configuration
    # Note: These are conservative settings for testing
    # Adjust based on your hardware capabilities
    config = PPOTrainingConfig(
        model_name="meta-llama/Meta-Llama-3.1-8B",
        use_4bit=True,  # Enable 4-bit quantization
        use_lora=True,  # Enable LoRA
        
        # Training settings (small for testing)
        batch_size=2,
        mini_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        max_steps=100,  # Small number for testing
        
        # PPO settings
        learning_rate=1.41e-5,
        ppo_epochs=4,
        
        # Generation settings
        max_new_tokens=64,  # Shorter for faster training
        
        # Logging
        log_with=None,  # Set to "wandb" to enable W&B logging
        logging_steps=5,
        save_steps=50,
        
        # Output
        output_dir="./ppo_llama_outputs",
        
        # Dataset
        max_dataset_samples=200,  # Small dataset for testing
    )
    
    print("\nTraining Configuration:")
    print("-"*80)
    print(f"Model: {config.model_name}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Max Steps: {config.max_steps}")
    print(f"Output Dir: {config.output_dir}")
    print(f"Using 4-bit: {config.use_4bit}")
    print(f"Using LoRA: {config.use_lora}")
    print("-"*80)
    
    # Initialize trainer
    trainer = LlamaPPOTrainer(config)
    
    try:
        # Run training
        trainer.run()
        
        print("\n" + "="*80)
        print("Training completed successfully!")
        print(f"Check outputs in: {config.output_dir}")
        print("="*80)
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"Error during training: {e}")
        print("="*80)
        print("\nCommon issues:")
        print("1. Insufficient GPU memory - try reducing batch_size")
        print("2. No access to Llama 3.1 - check HuggingFace authentication")
        print("3. Missing dependencies - check requirements")
        raise


if __name__ == "__main__":
    main()