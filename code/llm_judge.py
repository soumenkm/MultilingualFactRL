# ./code/llm_judge.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from typing import Tuple
import logging
from pathlib import Path
from datetime import datetime


class LLMJudge:
    """
    LLM-based judge using OpenAI GPT-4o-mini for answer evaluation.
    """
    
    def __init__(self, config: dict):
        """
        Initialize LLM judge.
        
        Args:
            config: Configuration dictionary with:
                - api_key: OpenAI API key
                - model_name: OpenAI model name (default: gpt-4o-mini)
                - temperature: Temperature for generation
                - judge_log_file: Path to save judgment logs
                - clear_judge_log: Whether to clear log file at start
        """
        self.logger = logging.getLogger(__name__)
        
        self.model_name = config.get("model_name", "gpt-4o-mini")
        api_key = config.get("api_key")
        
        if not api_key:
            raise ValueError("api_key must be provided in config")
        
        # Initialize OpenAI model
        self.llm = ChatOpenAI(
            model=self.model_name,
            openai_api_key=api_key,
            temperature=config.get("temperature", 0.0),
        )
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["generated_answer", "ground_truth", "aliases"],
            template="""You are an expert evaluator for question-answering systems. Your task is to determine if a generated answer matches the ground truth answer.

Ground Truth Answer: {ground_truth}
Acceptable Aliases: {aliases}
Generated Answer: {generated_answer}

Evaluation Guidelines:
1. Consider answers CORRECT if:
   - The core information matches (e.g., "3 years" vs "3.0 years")
   - Capitalization differs (e.g., "Paris" vs "paris")
   - Minor formatting differences (e.g., "11 years" vs "11.0 years")
   - The generated answer contains the correct answer even with extra text
   - Numbers match even if units differ slightly (e.g., "year" vs "years")
   - The answer matches any of the provided aliases

2. Consider answers INCORRECT if:
   - The factual information is wrong (e.g., "3 years" when answer is "11 years")
   - The entity/name is completely different
   - The number is significantly different

Answer ONLY with "CORRECT" or "INCORRECT" - nothing else.

Judgment:"""
        )
        
        # Setup judgment log file
        self.log_file = Path(config.get("judge_log_file", "./code/llm_judge_log.txt"))
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create/clear log file at start
        if config.get("clear_judge_log", True):
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("LLM JUDGE EVALUATION LOG\n")
                f.write(f"Model: {self.model_name}\n")
                f.write(f"Started at: {datetime.now().isoformat()}\n")
                f.write("="*80 + "\n\n")
        
        self.logger.info(f"Initialized LLM Judge with {self.model_name}")
        self.logger.info(f"LLM Judge logs will be saved to: {self.log_file}")
    
    def _save_judgment(
        self,
        prompt: str,
        response: str,
        generated: str,
        ground_truth: str,
        aliases: list,
        is_correct: bool
    ) -> None:
        """
        Save judgment to log file in human-readable format.
        
        Args:
            prompt: Full prompt sent to LLM
            response: LLM response
            generated: Generated answer (extracted)
            ground_truth: Ground truth answer
            aliases: List of aliases
            is_correct: Final judgment
        """
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-"*80 + "\n")
            f.write(f"Generated Answer: {generated}\n")
            f.write(f"Ground Truth:     {ground_truth}\n")
            f.write(f"Aliases:          {', '.join(aliases) if aliases else 'None'}\n")
            f.write("-"*80 + "\n")
            f.write(f"\nPROMPT SENT TO LLM:\n")
            f.write("-"*80 + "\n")
            f.write(f"{prompt}\n")
            f.write("-"*80 + "\n")
            f.write(f"\nLLM RESPONSE:\n")
            f.write("-"*80 + "\n")
            f.write(f"{response}\n")
            f.write("-"*80 + "\n")
            f.write(f"\nFINAL JUDGMENT: {'CORRECT' if is_correct else 'INCORRECT'}\n")
            f.write("="*80 + "\n\n")
    
    def judge(self, generated: str, ground_truth: str, aliases: list = None) -> bool:
        """
        Judge if generated answer matches ground truth.
        
        Args:
            generated: Generated answer
            ground_truth: Ground truth answer
            aliases: List of acceptable aliases
            
        Returns:
            True if correct, False otherwise
        """
        # Format aliases
        aliases_str = ", ".join(aliases) if aliases else "None"
        
        # Create prompt
        prompt = self.prompt_template.format(
            generated_answer=generated,
            ground_truth=ground_truth,
            aliases=aliases_str
        )
        
        try:
            # Call LLM
            response = self.llm.invoke(prompt)
            judgment_text = response.content.strip()
            
            # Parse judgment - FIXED LOGIC
            if judgment_text == "INCORRECT":
                is_correct = False
            elif judgment_text == "CORRECT":
                is_correct = True
            else:
                # Handle unexpected responses (e.g., "The answer is INCORRECT")
                self.logger.warning(f"Unexpected LLM response: '{judgment_text}'")
                # Fallback: check for INCORRECT first, then CORRECT
                if "INCORRECT" in judgment_text:
                    is_correct = False
                elif "CORRECT" in judgment_text:
                    is_correct = True
                else:
                    # If neither found, default to False and log
                    self.logger.error(f"Cannot parse judgment from: '{judgment_text}'")
                    is_correct = False
            
            # Save to log file
            self._save_judgment(
                prompt=prompt,
                response=judgment_text,
                generated=generated,
                ground_truth=ground_truth,
                aliases=aliases,
                is_correct=is_correct
            )
            
            self.logger.debug(
                f"Generated: {generated[:50]}... | GT: {ground_truth} | "
                f"Judgment: {judgment_text} | Correct: {is_correct}"
            )
            
            return is_correct
            
        except Exception as e:
            self.logger.error(f"Error in LLM judge: {e}")
            
            # Save error to log
            self._save_judgment(
                prompt=prompt,
                response=f"ERROR: {str(e)}",
                generated=generated,
                ground_truth=ground_truth,
                aliases=aliases,
                is_correct=False
            )
            
            # Fall back to False in case of error
            return False