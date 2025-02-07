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

@dataclass
class ModelConfig:
    name: str
    model_id: str
    provider: str
    client: Any

class TokenCounter:
    def __init__(self, model_name: str = 'gpt-4o'):
        self.encoding = tiktoken.encoding_for_model(model_name)
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))


class MetricsTracker:
    def __init__(self, project_name: str = 'SVx'):
        weave.init(project_name)
        self.token_counter = TokenCounter()
    
    def track_metrics(self, messages: List[Dict[str, str]], response: Dict) -> Dict:
        input_tokens = sum(self.token_counter.count_tokens(msg['content']) for msg in messages)
        output_text = response.choices[0].message.content
        output_tokens = self.token_counter.count_tokens(output_text)
        
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
        }

class LLMClient(ABC):
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.01) -> Dict:
        pass

class AISuiteClient(LLMClient):
    def __init__(self):
        import aisuite as ai
        self.client = ai.Client()
    
    def generate(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.01) -> Dict:
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=1500
        )

class TogetherClient(LLMClient):
    def __init__(self):
        from together import Together
        self.client = Together()
    
    def generate(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.01) -> Dict:
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=1500
        )

class OllamaClient(LLMClient):
    def __init__(self, host: str = "http://localhost:11434/v1/"):
        from openai import OpenAI
        self.client = OpenAI(
            base_url=host,
            api_key='ollama'
        )
    
    def generate(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.01) -> Dict:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=1500
        )
        return response

class vLLMClient(LLMClient):
    def __init__(self, endpoint: str = "http://localhost:8000"):
        import requests
        self.endpoint = endpoint
        
    def generate(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.01) -> Dict:
        response = requests.post(
            f"{self.endpoint}/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature
            }
        )
        return response.json()
