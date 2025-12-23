import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
from typing import List, Dict, Optional
import json
import os


class ModelEvaluator:
    """
    Class to evaluate trained PPO models.
    """
    
    def __init__(
        self,
        model_path: str,
        base_model_name: str = "meta-llama/Meta-Llama-3.1-8B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_4bit: bool = True
    ):
        """
        Initialize model evaluator.
        
        Args:
            model_path: Path to trained model (LoRA adapters or full model)
            base_model_name: Name of base model
            device: Device to run inference on
            load_in_4bit: Whether to load in 4-bit mode
        """
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        
        self.tokenizer = None
        self.model = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load trained model and tokenizer."""
        print(f"Loading model from {self.model_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Check if this is a LoRA adapter or full model
        adapter_config_path = os.path.join(self.model_path, "adapter_config.json")
        
        if os.path.exists(adapter_config_path):
            print("Detected LoRA adapters, loading base model + adapters...")
            
            # Load base model
            if self.load_in_4bit:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
            
            # Load LoRA adapters
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            print("LoRA adapters loaded successfully")
        else:
            print("Loading full model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate response for a given prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of responses to generate
            
        Returns:
            List of generated responses
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode
        responses = []
        for output in outputs:
            # Remove prompt from output
            response = self.tokenizer.decode(
                output[inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            responses.append(response)
        
        return responses
    
    def evaluate_on_prompts(
        self,
        prompts: List[str],
        save_results: bool = True,
        output_file: str = "evaluation_results.json"
    ) -> List[Dict]:
        """
        Evaluate model on a list of prompts.
        
        Args:
            prompts: List of prompts to evaluate
            save_results: Whether to save results to file
            output_file: Output file path
            
        Returns:
            List of evaluation results
        """
        print(f"\nEvaluating on {len(prompts)} prompts...")
        print("="*80)
        
        results = []
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}] Prompt:")
            print("-"*80)
            print(prompt)
            print("-"*80)
            
            # Generate response
            responses = self.generate_response(prompt)
            response = responses[0]
            
            print("Response:")
            print(response)
            print("="*80)
            
            # Store result
            result = {
                "prompt": prompt,
                "response": response,
                "prompt_length": len(prompt),
                "response_length": len(response),
            }
            results.append(result)
        
        # Save results
        if save_results:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {output_file}")
        
        return results
    
    def interactive_mode(self) -> None:
        """Run interactive evaluation mode."""
        print("\n" + "="*80)
        print("Interactive Evaluation Mode")
        print("="*80)
        print("Enter prompts to generate responses (type 'quit' to exit)")
        print("="*80 + "\n")
        
        while True:
            prompt = input("Prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Exiting interactive mode...")
                break
            
            if not prompt:
                continue
            
            print("\nGenerating response...")
            responses = self.generate_response(prompt)
            
            print("\nResponse:")
            print("-"*80)
            print(responses[0])
            print("-"*80 + "\n")
    
    def compare_with_base(
        self,
        prompt: str,
        max_new_tokens: int = 128
    ) -> Dict[str, str]:
        """
        Compare trained model with base model on same prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with both responses
        """
        print("\nGenerating response from trained model...")
        trained_response = self.generate_response(prompt, max_new_tokens=max_new_tokens)[0]
        
        # Would need to load base model separately for comparison
        # This is left as an exercise
        
        return {
            "prompt": prompt,
            "trained_model": trained_response,
        }


class EvaluationSuite:
    """
    Suite of evaluation prompts and metrics.
    """
    
    def __init__(self):
        """Initialize evaluation suite."""
        self.test_prompts = self._create_test_prompts()
    
    def _create_test_prompts(self) -> List[str]:
        """
        Create a diverse set of test prompts.
        
        Returns:
            List of test prompts
        """
        prompts = [
            "Review: This movie started with great promise. Continue this review in a positive tone:",
            "Review: The cinematography was stunning. Continue this review in a positive tone:",
            "Review: I had low expectations but. Continue this review in a positive tone:",
            "Review: The acting performances were. Continue this review in a positive tone:",
            "Review: Despite some flaws. Continue this review in a positive tone:",
        ]
        return prompts
    
    def get_prompts(self) -> List[str]:
        """
        Get evaluation prompts.
        
        Returns:
            List of prompts
        """
        return self.test_prompts


def main():
    """
    Main evaluation function.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate trained PPO model"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model (e.g., ./ppo_llama_outputs/final_model)"
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B",
        help="Base model name"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--test-suite",
        action="store_true",
        help="Run on test suite of prompts"
    )
    
    parser.add_argument(
        "--custom-prompt",
        type=str,
        help="Evaluate on a custom prompt"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for results"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("PPO Model Evaluation")
    print("="*80)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        base_model_name=args.base_model,
        load_in_4bit=True
    )
    
    # Run evaluation based on mode
    if args.interactive:
        evaluator.interactive_mode()
    
    elif args.test_suite:
        suite = EvaluationSuite()
        prompts = suite.get_prompts()
        evaluator.evaluate_on_prompts(
            prompts,
            save_results=True,
            output_file=args.output
        )
    
    elif args.custom_prompt:
        print(f"\nEvaluating custom prompt...")
        response = evaluator.generate_response(
            args.custom_prompt,
            max_new_tokens=args.max_tokens
        )[0]
        
        print("\n" + "="*80)
        print("Prompt:")
        print(args.custom_prompt)
        print("\nResponse:")
        print(response)
        print("="*80)
    
    else:
        print("\nNo evaluation mode specified. Use --help for options.")
        print("Available modes:")
        print("  --interactive: Interactive prompt mode")
        print("  --test-suite: Run on predefined test prompts")
        print("  --custom-prompt: Evaluate a single custom prompt")


if __name__ == "__main__":
    main()