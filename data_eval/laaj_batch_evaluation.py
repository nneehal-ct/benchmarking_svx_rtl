import pandas as pd
import json
from openai import OpenAI
from typing import List, Dict, Optional
import time
import weave
from dotenv import load_dotenv
import os 
from pydantic import BaseModel

# Load the variables from the .env file
load_dotenv()

ENV_VARS = {
    "WANDB_API_KEY": "WANDB_CASPIA_API_KEY",
    "OPENAI_API_KEY": "OPENAI_CASPIA_API_KEY",
    "GROQ_API_KEY": None,
    "HF_TOKEN": "HF_CASPIA_API_KEY",
    "TOGETHER_API_KEY": "TOGETHER_CASPIA_API_KEY",
    "ANTHROPIC_API_KEY": "ANTHROPIC_CASPIA_API_KEY"
}

for key, caspia_key in ENV_VARS.items():
    os.environ[key] = os.getenv(caspia_key or key, "")

# Initialize Weave Tracing
weave.init('SVx')

class ScoreDict(BaseModel):
    logical_equivalence: int
    signal_behavior: int
    edge_case_handling: int
    code_modularity: int
    resource_efficiency: int
    timing_pipeline_depth: int

def get_developer_prompt() -> str:
    # Prompt remains the same
    system_message = f""" 
    You are an expert in Verilog language and hardware design.
    Evaluate the following Verilog/SystemVerilog code against the ground truth based on these criteria:
    
    The evaluation criteria are as follows:
    - **Logical Equivalence** (0-5) : 
        - Think about the below points before making the output JSON object
        - Do both implementations produce identical outputs for all inputs?
        - Identify any functional differences.
        - Assign an integer **score (0-5)** value based on how closely the generated code matches the ground truth.
        
    - **Signal Behavior** (0-5):
        - Think about the below points before making the output JSON object
        - Do the registers, wires, and combinational logic behave the same way in both versions?
        - Are state transitions in FSMs identical?
        - Are signal updates happening at the correct time?
        - Assign an integer **score (0-5)** value based on how accurately the generated code preserves signal behavior.

    - **Edge Case Handling** (0-5):
        - Think about the below points before making the output JSON object
        - Does the generated code correctly implement synchronous and asynchronous resets?
        - Are there any potential issues with metastability, clock domain transfers, or data integrity?
        - Are unexpected behaviors prevented under corner cases?
        - Assign an integer **score (0-5)** value accordingly.

    - **Code Modularity** (0-5):
        - Think about the below points before making the output JSON object
        - Does the generated code follow modular design principles?
        - Are parameters, functions, and submodules effectively used?
        - Are unnecessary dependencies avoided?
        - Assign an integer **score (0-5)** value accordingly.

    - **Resource Efficiency** (0-5):
        - Think about the below points before making the output JSON object
        - Does the generated code use an excessive number of registers, combinational logic, or memory?
        - Are there inefficient resource utilization patterns?
        - Are logic and memory components minimized while maintaining functionality?
        - Assign an integer **score (0-5)** value accordingly.


    - **Timing & Pipeline Depth** (0-5):
        - Does the generated code introduce unnecessary delays?
        - Does it increase the critical path?
        - Are pipeline stages maintained or improved?
        - Assign an integer **score (0-5)** value.


    Return the scores exactly in the following **JSON format**:
    ```
    {{
      "logical_equivalence": X,
      "signal_behavior": X,
      "edge_case_handling": X,
      "code_modularity": X,
      "resource_efficiency": X,
      "timing_pipeline_depth": X
    }}
    ```
    For the scores, only mention the integer score value (0-5) and nothing else.
    Only return the score in JSON format. No need to explain anythings else. Do not include any other text or information in the response.
    """

def get_user_prompt(ground_truth: str, model_output: Optional[str], model_name: str) -> str:
    # Handle case where model output is NaN
    if pd.isna(model_output):
        return f"""
        **Ground Truth Code:**
        ```
        {ground_truth}
        ```

        **Generated Code:**
        ```
        No output available for model {model_name}
        ```
        
        Please evaluate this as a complete failure case with minimum scores.
        """
    
    return f"""
    **Ground Truth Code:**
    ```
    {ground_truth}
    ```

    **Generated Code:**
    ```
    {model_output}
    ```
    """

@weave.op()
def create_batch_request(ground_truth: str, model_output: Optional[str], model_name: str, index: int) -> Dict:
    """Create a single batch request with Weave tracking and NaN handling"""
    return {
        "custom_id": f"eval_{index}_{model_name}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": get_developer_prompt()
                },
                {
                    "role": "user",
                    "content": get_user_prompt(ground_truth, model_output, model_name)
                }
            ],
            "max_tokens": 1500,
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }
    }

@weave.op()
def validate_score_dict(response_dict: dict) -> ScoreDict:
    """Validate the response against ScoreDict model"""
    return ScoreDict(
        logical_equivalence=response_dict["logical_equivalence"],
        signal_behavior=response_dict["signal_behavior"],
        edge_case_handling=response_dict["edge_case_handling"],
        code_modularity=response_dict["code_modularity"],
        resource_efficiency=response_dict["resource_efficiency"],
        timing_pipeline_depth=response_dict["timing_pipeline_depth"]
    )

class BatchProcessor:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    def create_jsonl_for_evaluation(self, df: pd.DataFrame) -> str:
        """Convert dataframe to JSONL format for batch processing with NaN handling"""
        batch_requests = []
        
        for index, row in df.iterrows():
            ground_truth = row['ground_truth']
            
            # For each model column (excluding Ground_truth and Index)
            model_outputs = {col: row[col] for col in df.columns 
                           if col not in ['ground_truth', 'index']}
            
            # Create a prompt for each model output, handling NaN values
            for model_name, model_output in model_outputs.items():
                # Skip if both ground truth and model output are NaN
                if pd.isna(ground_truth) and pd.isna(model_output):
                    continue
                    
                request = create_batch_request(
                    ground_truth=ground_truth,
                    model_output=model_output,
                    model_name=model_name,
                    index=index
                )
                batch_requests.append(json.dumps(request))
        
        # Write to JSONL file
        jsonl_content = "\n".join(batch_requests)
        with open("batch_input.jsonl", "w") as f:
            f.write(jsonl_content)
        
        return "batch_input.jsonl"
    
    def process_batch_results(self, response_text: str) -> List[ScoreDict]:
        """Process and validate batch results"""
        results = []
        for line in response_text.splitlines():
            try:
                response = json.loads(line)
                validated_score = validate_score_dict(response)
                results.append(validated_score)
            except Exception as e:
                print(f"Error processing result: {e}")
                # Add a default score for failed processing
                default_score = ScoreDict(
                    logical_equivalence=0,
                    signal_behavior=0,
                    edge_case_handling=0,
                    code_modularity=0,
                    resource_efficiency=0,
                    timing_pipeline_depth=0
                )
                results.append(default_score)
        return results

    # Rest of the BatchProcessor class methods remain the same
    def upload_batch_file(self, input_file: str):
        return self.client.files.create(
            file=open(input_file, "rb"),
            purpose="batch"
        )

    def create_batch(self, file_id: str):
        return self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "HDL Code evaluation batch"
            }
        )

    def check_batch_status(self, batch_id: str):
        return self.client.batches.retrieve(batch_id)

    def save_batch_results(self, output_file_id: str, error_file_id: str = None):
        output = self.client.files.content(output_file_id)
        validated_results = self.process_batch_results(output.text)
        
        with open("batch_output.jsonl", "w") as f:
            for result in validated_results:
                f.write(json.dumps(result.model_dump()) + "\n")

    def process_batch(self, input_file: str) -> None:
        """Process a batch with proper error handling and file validation"""
        try:
            # Verify input file exists
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")

            # Upload file with error checking
            batch_file = self.upload_batch_file(input_file)
            if not batch_file or not batch_file.id:
                raise ValueError("File upload failed - no file ID received")
            print(f"File uploaded with ID: {batch_file.id}")

            # Create batch with error checking
            batch = self.create_batch(batch_file.id)
            if not batch or not batch.id:
                raise ValueError("Batch creation failed - no batch ID received")
            print(f"Batch created with ID: {batch.id}")
            
            # Monitor batch status
            while True:
                status = self.check_batch_status(batch.id)
                if not status:
                    raise ValueError("Failed to retrieve batch status")
                print(f"Status: {status.status}")
                
                if status.status in ['completed', 'failed', 'expired', 'cancelled']:
                    break
                    
                time.sleep(60)
            
            # Handle completion
            if status.status == 'completed':
                if not status.output_file_id:
                    raise ValueError("Completed batch has no output file ID")
                    
                self.save_batch_results(
                    output_file_id=status.output_file_id,
                    error_file_id=status.error_file_id
                )
                print("Results saved to batch_output.jsonl")
                if status.error_file_id:
                    print("Errors saved to batch_errors.jsonl")
            else:
                raise RuntimeError(f"Batch failed with status: {status.status}")
            
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            raise  # Re-raise the exception for proper error handling

def main():
    try:
        # Read CSV file
        df = pd.read_csv('test_batch_input.csv')
        print(f"Loaded {len(df)} rows from test_batch_input.csv")
        
        # Initialize processor
        processor = BatchProcessor()
        
        # Create and process batch
        input_file = processor.create_jsonl_for_evaluation(df)
        if not input_file:
            raise ValueError("Failed to create input JSONL file")
            
        print(f"Created batch input file: {input_file}")
        processor.process_batch(input_file)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise  # Re-raise for debugging

if __name__ == "__main__":
    main()