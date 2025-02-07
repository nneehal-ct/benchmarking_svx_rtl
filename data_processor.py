import os
import json
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
from dotenv import load_dotenv

load_dotenv()

ENV_VARS = {
    "WANDB_API_KEY": "WANDB_CASPIA_API_KEY",
    "OPENAI_API_KEY": None,
    "GROQ_API_KEY": None,
    "HF_TOKEN": "HF_CASPIA_API_KEY",
    "TOGETHER_API_KEY": "TOGETHER_API_KEY",
    "ANTHROPIC_API_KEY": "ANTHROPIC_CASPIA_API_KEY"
}

for key, caspia_key in ENV_VARS.items():
    os.environ[key] = os.getenv(caspia_key or key, "")

class DataProcessor:
    @staticmethod
    def load_dataset(dataset_name: str) -> pd.DataFrame:
        from datasets import load_dataset
        data = load_dataset(dataset_name)
        return data['train'].to_pandas()
    
    @staticmethod
    def parse_descriptions(df: pd.DataFrame) -> pd.DataFrame:
        df_parsed = df.copy()
        df_parsed['parsed_dict'] = df_parsed['parsed_description'].apply(lambda x: x if isinstance(x, dict) else json.loads(x))
        
        for field in ['block_summary', 'detailed_global_summary', 'high_level_global_summary']:
            df_parsed[f'system_{field}'] = df_parsed['parsed_dict'].apply(lambda x: x[field]['system'])
            df_parsed[f'prompt_{field}'] = df_parsed['parsed_dict'].apply(lambda x: x[field]['prompt'])
        
        df_parsed.drop('parsed_dict', axis=1, inplace=True)
        return df_parsed
    
    #from a dataframe select random n examples with seed and save as a separate CSV file
    @staticmethod
    def select_random_examples(df: pd.DataFrame, n: int, seed: int, save_path: str):
        df_sample = df.sample(n=n, random_state=seed)
        df_sample.to_csv(save_path, index=False)
        return df_sample
    
    #upload to huggingface hub 
    @staticmethod
    def push_to_huggingface_hub(df: pd.DataFrame, dataset_id: str):
        from datasets import Dataset
        dataset = Dataset.from_pandas(df)
        dataset.push_to_hub(dataset_id)
        return None 
    
# #If we just want to do the data parsing separately, we can do it like this:
# if __name__ == '__main__':
#     data = DataProcessor.load_dataset('caspia-technologies/benchmarking_rtl_svx')
#     print("Data loaded successfully")
#     df_sample = DataProcessor.select_random_examples(df=data, n=500, seed=42, save_path='final_data.csv')
#     print("Random examples selected and saved successfully")
#     DataProcessor.push_to_huggingface_hub(df=df_sample, dataset_id='caspia-technologies/benchmarking_rtl_svx_500_examples')
#     print("Data uploaded to Huggingface Hub successfully")