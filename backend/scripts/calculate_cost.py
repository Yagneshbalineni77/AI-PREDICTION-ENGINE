import os
import argparse
from google import genai
from google.genai import types

# Load environment using dotenv exactly like the main app to get LLM_API_KEY
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

# Fallback specifically for google-genai
if not os.environ.get("GEMINI_API_KEY") and os.environ.get("LLM_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.environ.get("LLM_API_KEY")

class GeminiCostCalculator:
    # Pricing per 1 Million Tokens (USD)
    PRICING = {
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    }

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.client = genai.Client()
        self.model = model_name
        self.prices = self.PRICING.get(model_name, self.PRICING["gemini-2.0-flash"])

    def count_tokens(self, text: str) -> int:
        """Uses the official google-genai library to precisely count tokens."""
        response = self.client.models.count_tokens(
            model=self.model,
            contents=[types.Content(role="user", parts=[types.Part(text=text)])]
        )
        return response.total_tokens

    def calculate_cost(self, input_tokens: int, output_tokens: int = 0) -> dict:
        """Calculates exact cost in USD based on input and output tokens."""
        input_cost = (input_tokens / 1_000_000) * self.prices["input"]
        output_cost = (output_tokens / 1_000_000) * self.prices["output"]
        total_cost = input_cost + output_cost
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }

def process_file(file_path: str, model: str):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
            
        print(f"\n--- Checking Cost for '{os.path.basename(file_path)}' ---")
        print(f"File Size: {len(text_content):,} characters")
        
        calculator = GeminiCostCalculator(model)
        print(f"Model selected: {model}")
        print("Connecting to Gemini API to count tokens...")
        
        token_count = calculator.count_tokens(text_content)
        costs = calculator.calculate_cost(input_tokens=token_count)
        
        print("\n=== Result ===")
        print(f"Tokens counted: {token_count:,}")
        print(f"Estimated Cost if used as Prompt: ${costs['total_cost']:.6f}")
        print("==============\n")
        
    except Exception as e:
        print(f"Error processing file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini API Token Cost Calculator")
    parser.add_argument("--file", type=str, required=True, help="Path to the text file to count")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help="Gemini model name")
    
    args = parser.parse_args()
    process_file(args.file, args.model)
