"""
Compute Cross-Lingual Consistency (CLC) and Transfer Scores on MKQA dataset.
Evaluates multilingual language models on factual recall across languages.
"""

import os
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from llm_judge import LLMJudge
from dotenv import load_dotenv
load_dotenv(dotenv_path="/home/soumen/api.env")
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from pathlib import Path


class MKQAEvaluator:
    """
    Class to evaluate cross-lingual consistency and transfer scores
    on MKQA dataset using a multilingual language model.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the MKQA evaluator.
        
        Args:
            config: Configuration dictionary containing:
                - model_name: HuggingFace model identifier
                - device: Device to run on (cuda/cpu)
                - max_samples: Maximum samples to evaluate (None for all)
                - languages: List of language codes to evaluate
                - batch_size: Batch size for generation
                - max_new_tokens: Maximum tokens to generate
                - temperature: Sampling temperature
                - top_k: Top-k sampling
                - top_p: Nucleus sampling
                - output_dir: Directory to save results
                - dataset_split: Which split to use (train/validation/test)
                - use_4bit: Whether to use 4-bit quantization
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Extract config parameters
        self.model_name = config.get("model_name", "google/gemma-2-2b")
        self.device = config.get("device", "cuda")
        self.max_samples = config.get("max_samples", None)
        self.languages = config.get("languages", ["en", "es", "fr", "de", "ja", "zh_cn"])
        self.batch_size = config.get("batch_size", 1)
        self.max_new_tokens = config.get("max_new_tokens", 50)
        self.temperature = config.get("temperature", 0.7)
        self.top_k = config.get("top_k", 50)
        self.top_p = config.get("top_p", 0.95)
        self.output_dir = config.get("output_dir", "./code/mkqa_results")
        self.use_4bit = config.get("use_4bit", True)
        
        # Initialize containers
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.results = defaultdict(list)
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized MKQAEvaluator with model: {self.model_name}")
        self.logger.info(f"Languages to evaluate: {self.languages}")
        self.logger.info(f"Max samples: {self.max_samples}")

        # Initialize LLM Judge
        judge_config = {
            "api_key": config.get("api_key"),
            "model_name": config.get("judge_model_name", "gemini-pro"),
            "temperature": 0.0,
            "judge_log_file": "./log/llm_judge_log.jsonl",
            "clear_judge_log": True,  # Clear log file at start of each run
        }
        self.llm_judge = LLMJudge(judge_config)
        self.logger.info("LLM Judge initialized")
    
    def _load_model(self) -> None:
        """Load the model and tokenizer from HuggingFace."""
        self.logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Setup quantization if enabled
        if self.use_4bit:
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
            self.logger.info("Model loaded with 4-bit quantization")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            self.logger.info("Model loaded in fp16")
        
        self.model.eval()
        self.logger.info("Model loaded successfully")
    
    def _load_dataset(self) -> None:
        """Load MKQA dataset."""
        self.logger.info("Loading MKQA dataset...")
        
        dataset_path = self.config.get("dataset_path")
        
        if dataset_path is None:
            self.logger.error("Dataset path not provided in config")
            raise ValueError("Please provide 'dataset_path' in config pointing to MKQA data")
        
        dataset_file = Path(dataset_path)
        
        if not dataset_file.exists():
            self.logger.error(f"Dataset file not found: {dataset_file}")
            raise FileNotFoundError(f"MKQA dataset not found at: {dataset_file}")
        
        self.logger.info(f"Loading dataset from: {dataset_file}")
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            self.dataset = [json.loads(line) for line in f]
        
        # Limit samples if specified
        if self.max_samples:
            self.dataset = self.dataset[:self.max_samples]
        
        self.logger.info(f"Loaded {len(self.dataset)} samples")
    
    def _create_prompt(self, question: str, language: str) -> str:
        """
        Create a prompt for the model.
        
        Args:
            question: The question text
            language: Language code
            
        Returns:
            Formatted prompt string
        """
        # Simple prompt template
        prompt = f"Question: {question}\nAnswer:"
        return prompt
    
    def _extract_answer_text(self, answer_obj: Dict) -> Optional[str]:
        """
        Extract answer text from MKQA answer object.
        
        Args:
            answer_obj: Answer object from MKQA
            
        Returns:
            Answer text or None
        """
        if not answer_obj:
            return None
        
        answer_type = answer_obj.get("type")
        
        if answer_type == "long_answer":
            # Long answer questions don't have short answers
            return None
        elif answer_type in ["number", "number_with_unit", "entity"]:
            return answer_obj.get("text", "").strip()
        
        return None
    
    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize answer for comparison.
        
        Args:
            answer: Raw answer text
            
        Returns:
            Normalized answer
        """
        if not answer:
            return ""
        
        # Convert to lowercase
        answer = answer.lower().strip()
        
        # Remove punctuation at the end
        while answer and answer[-1] in ".,!?;:":
            answer = answer[:-1]
        
        # Remove leading/trailing whitespace
        answer = answer.strip()
        
        return answer
    
    def _extract_answer(self, generated: str) -> str:
        """
        Extract answer from generated text.
        Takes only the first line before newlines.
        
        Args:
            generated: Generated text
            
        Returns:
            Extracted answer
        """
        # Split by double newline first
        if "\n\n" in generated:
            answer = generated.split("\n\n")[0]
        # Then by single newline
        elif "\n" in generated:
            answer = generated.split("\n")[0]
        else:
            answer = generated
        
        return answer.strip()
    
    def _check_answer_match(
        self,
        generated: str,
        ground_truth: str,
        aliases: List[str] = None
    ) -> bool:
        """
        Check if generated answer matches ground truth using LLM judge.
        
        Args:
            generated: Generated answer (full text)
            ground_truth: Ground truth answer
            aliases: List of acceptable aliases
            
        Returns:
            True if match, False otherwise
        """
        # Step 1: Extract answer (first line only)
        extracted_answer = self._extract_answer(generated)
        
        # Step 2: Use LLM as judge
        is_correct = self.llm_judge.judge(
            generated=extracted_answer,
            ground_truth=ground_truth,
            aliases=aliases
        )
        
        return is_correct
    
    def _generate_answer(self, prompt: str) -> str:
        """
        Generate answer for a given prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated answer text
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stop_strings=["\n\n", "\nQuestion:", "\n\n\n", "\n"],  # Stop at these sequences
                tokenizer=self.tokenizer,  # Need tokenizer for stop_strings
            )
        
        # Decode only the generated part
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return answer.strip()
    
    def _evaluate_sample(self, sample: Dict) -> Dict:
        """
        Evaluate a single sample across all languages.
        
        Args:
            sample: MKQA sample
            
        Returns:
            Dictionary with results for this sample
        """
        queries = sample.get("queries", {})
        answers = sample.get("answers", {})
        
        sample_results = {
            "example_id": sample.get("example_id"),
            "language_results": {},
            "correct_languages": [],
        }
        
        # Evaluate each language
        for lang in self.languages:
            if lang not in queries or lang not in answers:
                self.logger.warning(f"Language {lang} not found in sample")
                continue
            
            query = queries[lang]
            answer_list = answers[lang]
            
            # Skip if no answer
            if not answer_list or answer_list[0] is None:
                continue
            
            answer_obj = answer_list[0]
            ground_truth = self._extract_answer_text(answer_obj)
            
            # Skip long answer questions
            if ground_truth is None:
                continue
            
            # Create prompt and generate
            prompt = self._create_prompt(query, lang)
            generated = self._generate_answer(prompt)
            
            # Check correctness
            aliases = answer_obj.get("aliases", [])
            is_correct = self._check_answer_match(generated, ground_truth, aliases)
            
            sample_results["language_results"][lang] = {
                "query": query,
                "generated": generated,
                "ground_truth": ground_truth,
                "correct": is_correct,
            }
            
            if is_correct:
                sample_results["correct_languages"].append(lang)
        
        return sample_results
    
    def _compute_clc(self, all_results: List[Dict]) -> Dict:
        """
        Compute Cross-Lingual Consistency scores.
        
        Args:
            all_results: List of per-sample results
            
        Returns:
            Dictionary with CLC scores
        """
        self.logger.info("Computing Cross-Lingual Consistency (CLC) scores...")
        
        clc_scores = []
        per_language_accuracy = defaultdict(list)
        
        for result in all_results:
            num_languages = len(result["language_results"])
            if num_languages == 0:
                continue
            
            num_correct = len(result["correct_languages"])
            clc_score = num_correct / num_languages
            clc_scores.append(clc_score)
            
            # Track per-language accuracy
            for lang, lang_result in result["language_results"].items():
                per_language_accuracy[lang].append(1.0 if lang_result["correct"] else 0.0)
        
        # Compute statistics
        mean_clc = np.mean(clc_scores) if clc_scores else 0.0
        std_clc = np.std(clc_scores) if clc_scores else 0.0
        
        # Per-language accuracy
        lang_accuracy = {
            lang: np.mean(scores) for lang, scores in per_language_accuracy.items()
        }
        
        clc_results = {
            "mean_clc": mean_clc,
            "std_clc": std_clc,
            "num_samples": len(clc_scores),
            "per_language_accuracy": lang_accuracy,
        }
        
        self.logger.info(f"Mean CLC: {mean_clc:.4f} ± {std_clc:.4f}")
        
        return clc_results
    
    def _compute_transfer_scores(self, all_results: List[Dict]) -> Dict:
        """
        Compute Transfer scores between language pairs.
        
        Args:
            all_results: List of per-sample results
            
        Returns:
            Dictionary with transfer scores
        """
        self.logger.info("Computing Transfer scores...")
        
        transfer_scores = defaultdict(lambda: {"correct_both": 0, "source_correct": 0})
        
        for result in all_results:
            correct_langs = set(result["correct_languages"])
            all_langs = set(result["language_results"].keys())
            
            # Compute transfer for all pairs
            for source_lang in all_langs:
                source_correct = source_lang in correct_langs
                
                if not source_correct:
                    continue
                
                for target_lang in all_langs:
                    if source_lang == target_lang:
                        continue
                    
                    target_correct = target_lang in correct_langs
                    
                    pair_key = f"{source_lang}->{target_lang}"
                    transfer_scores[pair_key]["source_correct"] += 1
                    
                    if target_correct:
                        transfer_scores[pair_key]["correct_both"] += 1
        
        # Compute transfer percentages
        transfer_results = {}
        for pair, counts in transfer_scores.items():
            if counts["source_correct"] > 0:
                transfer_score = counts["correct_both"] / counts["source_correct"]
                transfer_results[pair] = {
                    "transfer_score": transfer_score,
                    "correct_both": counts["correct_both"],
                    "source_correct": counts["source_correct"],
                }
        
        # Compute average transfer score
        all_transfer_scores = [v["transfer_score"] for v in transfer_results.values()]
        mean_transfer = np.mean(all_transfer_scores) if all_transfer_scores else 0.0
        
        self.logger.info(f"Mean Transfer Score: {mean_transfer:.4f}")
        
        return {
            "mean_transfer_score": mean_transfer,
            "pair_transfer_scores": transfer_results,
        }
    
    def evaluate(self) -> Dict:
        """
        Run full evaluation pipeline.
        
        Returns:
            Dictionary with all evaluation results
        """
        self.logger.info("="*80)
        self.logger.info("Starting MKQA Evaluation")
        self.logger.info("="*80)
        
        # Load model and dataset
        self._load_model()
        self._load_dataset()
        
        # Evaluate all samples
        self.logger.info(f"Evaluating {len(self.dataset)} samples...")
        all_results = []
        
        for sample in tqdm(self.dataset, desc="Evaluating samples"):
            result = self._evaluate_sample(sample)
            all_results.append(result)
        
        # Compute metrics
        clc_results = self._compute_clc(all_results)
        transfer_results = self._compute_transfer_scores(all_results)
        
        # Combine results
        final_results = {
            "model_name": self.model_name,
            "num_samples": len(all_results),
            "languages": self.languages,
            "clc_results": clc_results,
            "transfer_results": transfer_results,
            "detailed_results": all_results,
        }
        
        # Save results
        self._save_results(final_results)
        
        self.logger.info("="*80)
        self.logger.info("Evaluation Complete")
        self.logger.info("="*80)
        
        return final_results
    
    def _save_results(self, results: Dict) -> None:
        """
        Save results to JSON file.
        
        Args:
            results: Results dictionary
        """
        output_dir = Path(self.output_dir)
        output_path = output_dir / "evaluation_results.json"
        
        # Remove detailed results for summary file
        summary_results = {k: v for k, v in results.items() if k != "detailed_results"}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Summary results saved to: {output_path}")
        
        # Save detailed results separately
        detailed_path = output_dir / "detailed_results.json"
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(results["detailed_results"], f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Detailed results saved to: {detailed_path}")
    
    def print_summary(self, results: Dict) -> None:
        """
        Print summary of results.
        
        Args:
            results: Results dictionary
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("="*80)
        
        # CLC results
        clc = results["clc_results"]
        self.logger.info(f"\nCross-Lingual Consistency (CLC):")
        self.logger.info(f"  Mean CLC: {clc['mean_clc']:.4f} ± {clc['std_clc']:.4f}")
        self.logger.info(f"  Number of samples: {clc['num_samples']}")
        
        self.logger.info(f"\nPer-Language Accuracy:")
        for lang, acc in sorted(clc['per_language_accuracy'].items()):
            self.logger.info(f"  {lang}: {acc:.4f}")
        
        # Transfer results
        transfer = results["transfer_results"]
        self.logger.info(f"\nTransfer Scores:")
        self.logger.info(f"  Mean Transfer Score: {transfer['mean_transfer_score']:.4f}")
        
        # Show top and bottom transfer pairs
        pairs = sorted(
            transfer['pair_transfer_scores'].items(),
            key=lambda x: x[1]['transfer_score'],
            reverse=True
        )
        
        self.logger.info(f"\nTop 10 Transfer Pairs:")
        for pair, scores in pairs[:10]:
            self.logger.info(
                f"  {pair}: {scores['transfer_score']:.4f} "
                f"({scores['correct_both']}/{scores['source_correct']})"
            )
        
        self.logger.info(f"\nBottom 10 Transfer Pairs:")
        for pair, scores in pairs[-10:]:
            self.logger.info(
                f"  {pair}: {scores['transfer_score']:.4f} "
                f"({scores['correct_both']}/{scores['source_correct']})"
            )
        
        self.logger.info("="*80)


def main():
    """Main function to run evaluation."""
    # Setup logging
    log_file = Path("./log") / "mkqa_evaluation.log"
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
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        device = "cuda"
    else:
        logger.warning("GPU not available, using CPU")
        device = "cpu"
    
    # Configuration dictionary
    config = {
        # Model settings
        "model_name": "google/gemma-2-2b",  # Gemma 2B model
        "device": device,
        "use_4bit": False,  # Use 4-bit quantization to save memory
        
        # Dataset settings
        "dataset_path": "./data/mkqa.jsonl",  # Path to MKQA data
        "max_samples": 10,  # Start with 10 samples for testing, set to None for full dataset
        
        # Languages to evaluate (subset of MKQA's 26 languages)
        "languages": [
            "en",     # English
            "es",     # Spanish
            "fr",     # French
            "de",     # German
            "ja",     # Japanese
            "zh_cn",  # Chinese (Simplified)
            "ar",     # Arabic
            "ru",     # Russian
        ],
        
        # Generation settings
        "batch_size": 1,
        "max_new_tokens": 50,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
        
        # Output settings
        "output_dir": "./output/mkqa_results",
        
        # Add this new parameter
        "api_key": os.getenv("OPENAI"),  # Get from Google AI Studio
        "judge_model_name": "gpt-4o-mini",  # Or "gemini-1.5-pro" for better quality
    
    }
    logger.info("="*80)
    logger.info("MKQA Cross-Lingual Evaluation")
    logger.info("="*80)
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("="*80)
    
    try:
        # Initialize evaluator
        evaluator = MKQAEvaluator(config)
        
        # Run evaluation
        results = evaluator.evaluate()
        
        # Print summary
        evaluator.print_summary(results)
        
        logger.info("\nEvaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()