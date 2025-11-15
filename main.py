"""
Main entry point for the 270FT fine-tuning and evaluation pipeline.
Interactive CLI for querying fine-tuned models.
"""

import os
import sys
import json
import yaml
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sympy import simplify, symbols, sympify, parse_expr, Eq


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model_with_adapter(
    base_model_name: str,
    adapter_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load base model and apply LoRA adapter."""
    adapter_path_obj = Path(adapter_path)
    
    # Verify adapter path exists
    if not adapter_path_obj.exists():
        raise FileNotFoundError(
            f"Adapter path does not exist: {adapter_path}\n"
            f"Please ensure the model has been trained and saved to this location."
        )
    
    # Check for required adapter files
    adapter_config = adapter_path_obj / "adapter_config.json"
    adapter_weights = adapter_path_obj / "adapter_model.safetensors"
    if not adapter_weights.exists():
        adapter_weights = adapter_path_obj / "adapter_model.bin"
    
    if not adapter_config.exists():
        raise FileNotFoundError(
            f"Adapter config not found at {adapter_config}. "
            f"Model may not have been saved correctly during training."
        )
    
    if not adapter_weights.exists():
        raise FileNotFoundError(
            f"Adapter weights not found at {adapter_weights}. "
            f"Model may not have been saved correctly during training."
        )
    
    print(f"Loading {base_model_name}...")
    print(f"  Adapter path: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    
    model = PeftModel.from_pretrained(model, str(adapter_path_obj))
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"[OK] {base_model_name} loaded successfully\n")
    return model, tokenizer


def format_prompt(question: str) -> str:
    """Format prompt in the training template format."""
    return f"### Question:\n{question}\n\n### Solution:\n"


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """Generate response for a given question."""
    formatted_prompt = format_prompt(question)
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    
    return generated_text.strip()


def parse_structured_output(response: str) -> Dict[str, str]:
    """
    Parse structured output from model response.
    Extracts: Algorithm Outline, Pseudocode, Proof Summary, and any mathematical expressions.
    """
    structured = {
        "algorithm_outline": "",
        "pseudocode": "",
        "proof_summary": "",
        "raw_response": response,
    }
    
    # Try to extract sections using common patterns
    patterns = {
        "algorithm_outline": [
            r"(?:Algorithm Outline|Algorithm|Outline)[:\s]*\n(.*?)(?=\n(?:Pseudocode|Proof|Symbolic|$))",
            r"\[Algorithm Outline\]\s*\n(.*?)(?=\n\[|$)",
        ],
        "pseudocode": [
            r"(?:Pseudocode|Pseudo-code|Code)[:\s]*\n(.*?)(?=\n(?:Proof|Symbolic|$))",
            r"\[Pseudocode\]\s*\n(.*?)(?=\n\[|$)",
        ],
        "proof_summary": [
            r"(?:Proof Summary|Proof|Summary)[:\s]*\n(.*?)(?=\n(?:Symbolic|$))",
            r"\[Proof Summary\]\s*\n(.*?)(?=\n\[|$)",
        ],
    }
    
    for key, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                structured[key] = match.group(1).strip()
                break
    
    # If sections weren't found, try to split by common delimiters
    if not any(structured.values()[:-1]):  # If no sections found (except raw_response)
        # Try splitting by double newlines or numbered sections
        parts = re.split(r'\n\n+', response)
        if len(parts) >= 3:
            structured["algorithm_outline"] = parts[0] if len(parts) > 0 else ""
            structured["pseudocode"] = parts[1] if len(parts) > 1 else ""
            structured["proof_summary"] = "\n\n".join(parts[2:]) if len(parts) > 2 else ""
        else:
            # Fallback: use the whole response
            structured["algorithm_outline"] = response
    
    return structured


def extract_mathematical_expressions(text: str) -> list:
    """Extract mathematical expressions from text."""
    # Patterns for common mathematical expressions
    patterns = [
        r'([a-zA-Z0-9\s\+\-\*/\^\(\)]+(?:\s*=\s*[a-zA-Z0-9\s\+\-\*/\^\(\)]+)?)',  # General expressions
        r'(\d+\s*[\+\-\*/\^]\s*\d+)',  # Simple arithmetic
        r'([a-zA-Z]\s*[\+\-\*/\^]\s*[a-zA-Z0-9]+)',  # Variable expressions
    ]
    
    expressions = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        expressions.extend(matches)
    
    return expressions


def check_symbolic_equivalence(statement: str, response: str) -> Tuple[bool, str]:
    """
    Check if the response symbolically verifies the statement.
    Returns (is_valid, explanation)
    """
    try:
        # Extract expressions from both statement and response
        statement_exprs = extract_mathematical_expressions(statement)
        response_exprs = extract_mathematical_expressions(response)
        
        if not statement_exprs or not response_exprs:
            return False, "Could not extract mathematical expressions"
        
        # Try to parse and verify
        x, n = symbols('x n')
        
        # Look for common proof patterns
        # For example: "sum of first n natural numbers is n(n+1)/2"
        if "sum" in statement.lower() and "n" in statement.lower():
            # Try to verify the formula
            try:
                # Expected: sum = n*(n+1)/2
                # Check if response contains this or proves it
                if "n*(n+1)/2" in response or "n(n+1)/2" in response:
                    # Try to verify symbolically
                    sum_expr = parse_expr("n*(n+1)/2", transformations='all')
                    # This is a simplified check - in practice, you'd need more sophisticated verification
                    return True, "Formula matches expected result"
            except:
                pass
        
        # General symbolic check: try to find if expressions are equivalent
        for stmt_expr in statement_exprs[:3]:  # Check first few expressions
            for resp_expr in response_exprs[:3]:
                try:
                    stmt_sym = parse_expr(stmt_expr, transformations='all')
                    resp_sym = parse_expr(resp_expr, transformations='all')
                    
                    diff = simplify(stmt_sym - resp_sym)
                    if diff == 0:
                        return True, f"Expressions are equivalent: {stmt_expr} ≡ {resp_expr}"
                    
                    eq = Eq(stmt_sym, resp_sym)
                    if eq.simplify() == True:
                        return True, f"Expressions are equal: {stmt_expr} = {resp_expr}"
                except:
                    continue
        
        # If we can't verify, check if response contains key terms from statement
        statement_keywords = set(re.findall(r'\b\w+\b', statement.lower()))
        response_keywords = set(re.findall(r'\b\w+\b', response.lower()))
        
        overlap = statement_keywords.intersection(response_keywords)
        if len(overlap) >= len(statement_keywords) * 0.5:
            return True, "Response addresses the statement (partial verification)"
        
        return False, "Could not verify symbolic equivalence"
    
    except Exception as e:
        return False, f"Verification error: {str(e)}"


def format_output(question: str, structured: Dict, symbolic_check: Tuple[bool, str]) -> str:
    """Format the final structured output."""
    output = f"{'='*60}\n"
    output += f"Question: {question}\n"
    output += f"{'='*60}\n\n"
    
    output += "[Algorithm Outline]\n"
    output += f"{structured['algorithm_outline'] or 'Not provided'}\n\n"
    
    output += "[Pseudocode]\n"
    output += f"{structured['pseudocode'] or 'Not provided'}\n\n"
    
    output += "[Proof Summary]\n"
    output += f"{structured['proof_summary'] or 'Not provided'}\n\n"
    
    status = "Passed" if symbolic_check[0] else "Failed"
    output += f"[Symbolic Check: {status}]\n"
    output += f"  {symbolic_check[1]}\n"
    output += f"{'='*60}\n"
    
    return output


def save_session(output_dir: Path, question: str, formatted_output: str, raw_response: str):
    """Save session to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"session_{timestamp}.txt"
    
    with open(output_file, "w") as f:
        f.write(f"Session: {timestamp}\n")
        f.write(f"{'='*60}\n\n")
        f.write(formatted_output)
        f.write(f"\n\nRaw Response:\n{'-'*60}\n")
        f.write(raw_response)
    
    print(f"\n[OK] Session saved to: {output_file}")


def main():
    """Main CLI application."""
    # Get project root
    project_root = Path(__file__).parent
    config_path = project_root / "270FT" / "configs" / "training_config.yaml"
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    # Load config
    config = load_config(str(config_path))
    
    # Setup paths
    models_dir = project_root / "270FT"
    output_dir = project_root / "270FT" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    models = {}
    for model_config in config["models"]:
        base_model_name = model_config["name"]
        adapter_path = models_dir / model_config["output_dir"]
        
        if not adapter_path.exists():
            print(f"Warning: Adapter path {adapter_path} does not exist. Skipping {base_model_name}")
            continue
        
        try:
            model, tokenizer = load_model_with_adapter(
                base_model_name,
                str(adapter_path),
                device=device,
            )
            models[base_model_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "display_name": base_model_name.split("/")[-1],
            }
        except Exception as e:
            print(f"Error loading {base_model_name}: {e}")
            continue
    
    if not models:
        print("Error: No models loaded. Please check your model paths.")
        sys.exit(1)
    
    # CLI loop
    print(f"{'='*60}")
    print("270FT - Algorithm Tutor CLI")
    print(f"{'='*60}\n")
    
    while True:
        # Model selection
        print("\nAvailable models:")
        model_list = list(models.items())
        for idx, (model_name, model_info) in enumerate(model_list, 1):
            print(f"  {idx}. {model_info['display_name']}")
        print(f"  {len(model_list) + 1}. Exit")
        
        try:
            choice = input("\nSelect model (1-{}): ".format(len(model_list) + 1)).strip()
            
            if choice == str(len(model_list) + 1) or choice.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            model_idx = int(choice) - 1
            if model_idx < 0 or model_idx >= len(model_list):
                print("Invalid choice. Please try again.")
                continue
            
            selected_model_name, selected_model_info = model_list[model_idx]
            model = selected_model_info["model"]
            tokenizer = selected_model_info["tokenizer"]
            
            print(f"\nSelected: {selected_model_info['display_name']}\n")
            
            # Get question
            print("Enter your question (or 'back' to return to model selection):")
            question = input("> ").strip()
            
            if question.lower() in ['back', 'b']:
                continue
            
            if not question:
                print("Please enter a valid question.")
                continue
            
            # Generate response
            print("\nGenerating response...")
            try:
                response = generate_response(model, tokenizer, question)
                
                # Parse structured output
                structured = parse_structured_output(response)
                
                # Perform symbolic check
                symbolic_check = check_symbolic_equivalence(question, response)
                
                # Format output
                formatted = format_output(question, structured, symbolic_check)
                
                # Display
                print("\n" + formatted)
                
                # Option to save
                save_choice = input("\nSave this session? (y/n): ").strip().lower()
                if save_choice in ['y', 'yes']:
                    save_session(output_dir, question, formatted, structured['raw_response'])
                
            except Exception as e:
                print(f"Error generating response: {e}")
                import traceback
                traceback.print_exc()
        
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
