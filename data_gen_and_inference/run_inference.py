import os
import yaml
import pandas as pd
import numpy as np
import tiktoken
import weave
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from abc import ABC, abstractmethod

from data_processor import DataProcessor
from client_processor import LLMClient, AISuiteClient, TogetherClient, OllamaClient, vLLMClient, MetricsTracker, TokenCounter, ModelConfig

from dotenv import load_dotenv
load_dotenv()

ENV_VARS = {
    "WANDB_API_KEY": "WANDB_CASPIA_API_KEY",
    "OPENAI_API_KEY": "OPENAI_CASPIA_API_KEY",
    "GROQ_API_KEY": None,
    "HF_TOKEN": "HF_API_KEY",
    "TOGETHER_API_KEY": "TOGETHER_CASPIA_API_KEY",
    "ANTHROPIC_API_KEY": "ANTHROPIC_CASPIA_API_KEY"
}

for key, caspia_key in ENV_VARS.items():
    os.environ[key] = os.getenv(caspia_key or key, "")


class InferencePipeline:
    def __init__(
        self,
        models_config: str,
        dataset_name: str,
        metrics_tracker: Optional[MetricsTracker] = None
    ):
        # Load YAML configuration file
        with open(models_config, 'r') as f:
            self.models = yaml.safe_load(f)
        
        self.dataset_name = dataset_name
        self.metrics_tracker = metrics_tracker or MetricsTracker()
        self.processor = DataProcessor()
        
        self.clients = {
            "aisuite": AISuiteClient(),
            "together": TogetherClient(),
            "ollama": OllamaClient(),
        }
    
    def _format_messages(self, query: Dict, examples: List[Dict]) -> List[Dict[str, str]]:
        system_template = """You are a helpful assistant. You will be given a query along with two example queries and corresponding answers to follow.
        Based on that, you will need to provide a solution to the query. Only answer with Verilog code block, do not generate anything else. Do not give any explanation or context."""
        
        examples_text = ""
        for i, (example, code) in enumerate(examples, start=1):
            examples_text += f"\n\n<Example {i}>\n##Instruction: {example['system_block_summary']}"
            examples_text += f"\n##Query: {example['prompt_block_summary']} \n More Global Details: {example['prompt_detailed_global_summary']}"
            examples_text += f"\n##Answer: {code}\n</Example {i}>\n\n"
        
        user_query = f"<User Query> {query['system_block_summary']} {query['prompt_block_summary']} \n More Global Details: {example['prompt_detailed_global_summary']} </User Query>"
        
        return [
            {"role": "system", "content": system_template + examples_text},
            {"role": "user", "content": user_query}
        ]
    
    @weave.op()
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception)
    )
    def _generate_response(self, client: LLMClient, model: str, messages: List[Dict[str, str]]) -> Dict:
        try:
            response = client.generate(messages, model)
            metrics = self.metrics_tracker.track_metrics(messages, response)
            return {
                'response': response.choices[0].message.content,
                'metrics': metrics
            }
        except Exception as e:
            if "rate limit" in str(e).lower():
                print(f"Rate limit hit for {model}. Saving progress and stopping...")
                raise RuntimeError("Rate limit exceeded")
            print(f"Error generating response for {model}: {str(e)}")
            return {
                'response': None,
                'metrics': None
            }
    
    def run_inference(self, limit: Optional[int] = None, after_index: Optional[int] = None) -> pd.DataFrame:
        df = self.processor.load_dataset(self.dataset_name)
        df_parsed = self.processor.parse_descriptions(df)
        results = []
        
        try:
            for idx, row in df_parsed.iterrows():
                if limit and idx >= limit:
                    break

                if after_index and idx < after_index:
                    print(f"Skipping row {idx+1}")
                    continue
                    
                try:
                    examples = [(df_parsed.iloc[0], df_parsed.iloc[0]['code']),
                              (df_parsed.iloc[1], df_parsed.iloc[1]['code'])]
                    
                    messages = self._format_messages(row, examples)
                    row_results = {'index': idx, 'ground_truth': row['code']}
                    
                    for provider, config in self.models.items():
                        client = self.clients[provider]
                        for model_name, model_id in config['models'].items():
                            try:
                                response = self._generate_response(client, model_id, messages)
                                row_results[f"{provider}_{model_name}"] = response['response']
                            except RuntimeError:
                                # Save current results and exit
                                df_results = pd.DataFrame(results)
                                df_results.to_csv('results.csv', index=False)
                                return df_results
                            except Exception as e:
                                print(f"Error processing {provider}_{model_name}: {e}")
                                row_results[f"{provider}_{model_name}"] = None
                    
                    results.append(row_results)
                    print(f"Processed row {idx+1}")
                    
                    # Save progress after each row
                    df_results = pd.DataFrame(results)
                    df_results.to_csv('results.csv', index=False)
                    
                except Exception as e:
                    print(f"Error processing row {idx}: {e}")
                    continue

        except KeyboardInterrupt:
            print("\nSaving progress before exit...")
            
        finally:
            df_results = pd.DataFrame(results)
            df_results.to_csv('results.csv', index=False)
            
        return df_results
    
if __name__ == "__main__":
    models_config = "models.yaml"  # Changed from models.json to models.yaml
    dataset_name = "caspia-technologies/benchmarking_rtl_svx_500_examples"
    
    pipeline = InferencePipeline(
        models_config=models_config,
        dataset_name=dataset_name
    )
    
    results = pipeline.run_inference(after_index=30)
    print(f"Processed {len(results)} examples")