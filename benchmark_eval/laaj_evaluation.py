from openai import OpenAI
from openai import OpenAIError, RateLimitError
import pandas as pd
import json
import re
import os
import weave
import math
import time
import signal
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv
from pydantic import BaseModel

# Global variable to track if the process should be interrupted
should_interrupt = False

def signal_handler(signum, frame):
    global should_interrupt
    print("\nInterrupt signal received. Saving progress and exiting gracefully...")
    should_interrupt = True

# Register the signal handler for CTRL+C
signal.signal(signal.SIGINT, signal_handler)

# Load the variables from the .env file
load_dotenv()

ENV_VARS = {
    "WANDB_API_KEY": "WANDB_CASPIA_API_KEY",
    "OPENAI_API_KEY": "OPENAI_API_KEY",
    "GROQ_API_KEY": None,
    "HF_TOKEN": "HF_CASPIA_API_KEY",
    "TOGETHER_API_KEY": "TOGETHER_API_KEY",
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

def save_progress(df, suffix="checkpoint"):
    """Save the current progress to a CSV file with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/benchmarking_nl_to_rtl/results_{suffix}_{timestamp}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Progress saved to {filename}")
    return filename

@weave.op()
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((RateLimitError, OpenAIError))
)
def benchmark_laaj_openai_response(system_message, prompt):
    client = OpenAI()
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "developer", "content": system_message},
                 {"role": "user", "content": prompt}],
        temperature=0,
        response_format=ScoreDict
    )
    return response.choices[0].message.parsed

def get_messages(ground_truth, generated_code):

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

    prompt = f"""
    **Ground Truth Code:**
    ```
    {ground_truth}
    ```

    **Generated Code:**
    ```
    {generated_code}
    ```
    """

    return system_message, prompt

def evaluate_verilog_code(ground_truth, generated_code):
    if generated_code == "":
        return {
            "logical_equivalence": 0,
            "signal_behavior": 0,
            "edge_case_handling": 0,
            "code_modularity": 0,
            "resource_efficiency": 0,
            "timing_pipeline_depth": 0
        }

    system_message, prompt = get_messages(ground_truth, generated_code)

    try:
        result = benchmark_laaj_openai_response(system_message, prompt)
        return result.model_dump(mode="json")
    except Exception as e:
        print(f"Error in evaluate_verilog_code: {e}")
        return {
            "logical_equivalence": 0,
            "signal_behavior": 0,
            "edge_case_handling": 0,
            "code_modularity": 0,
            "resource_efficiency": 0,
            "timing_pipeline_depth": 0
        }

def compute_scaled_score(logical_equivalence, signal_behavior, edge_case_handling,
                        code_modularity, resource_efficiency, timing_pipeline_depth):
    weights = {
        "Logical_Score": 0.50,
        "Signal_Score": 0.10,
        "EdgeCase_Score": 0.10,
        "Modularity_Score": 0.10,
        "ResourceEfficiency_Score": 0.10,
        "Timing_Score": 0.10
    }
    
    weighted_sum = (
        logical_equivalence * weights["Logical_Score"] +
        signal_behavior * weights["Signal_Score"] +
        edge_case_handling * weights["EdgeCase_Score"] +
        code_modularity * weights["Modularity_Score"] +
        resource_efficiency * weights["ResourceEfficiency_Score"] +
        timing_pipeline_depth * weights["Timing_Score"]
    )
    
    max_score = 5.0
    max_weighted_sum = sum(weight * max_score for weight in weights.values())
    scaled_score = weighted_sum / max_weighted_sum
    scaled_score = math.ceil(scaled_score * 100) / 100.0
    
    return scaled_score

def make_json(ground_truth, generated_code):
    try:
        scores = evaluate_verilog_code(ground_truth, generated_code)
        #print(f"Scores received: {scores}")

        scaled_score = compute_scaled_score(
            scores["logical_equivalence"],
            scores["signal_behavior"],
            scores["edge_case_handling"],
            scores["code_modularity"],
            scores["resource_efficiency"],
            scores["timing_pipeline_depth"]
        )
        
        score_json = {
            "scaled_score": scaled_score,
            **scores
        }

        return json.dumps(score_json)
    except Exception as e:
        print(f"Error in make_json: {e}")
        return json.dumps({
            "scaled_score": 0,
            "logical_equivalence": 0,
            "signal_behavior": 0,
            "edge_case_handling": 0,
            "code_modularity": 0,
            "resource_efficiency": 0,
            "timing_pipeline_depth": 0
        })

def add_score_column(df_source, df_target, col1, col2, new_col_name):
    """
    Process each row and add scores to the target dataframe with error handling and progress saving
    """
    total_rows = len(df_source)
    
    try:
        for idx, row in df_source.iterrows():
            if should_interrupt:
                print(f"Processing interrupted at row {idx}/{total_rows}")
                save_progress(df_target, "interrupt_backup")
                return df_target
                
            if df_target.get(new_col_name) is None:
                df_target[new_col_name] = pd.Series()
            df_target.loc[idx, new_col_name] = make_json(row[col1], row[col2])
            
            if idx > 0 and idx % 5 == 0:
                print(f"Processed {idx}/{total_rows} rows...")
                save_progress(df_target, "intermediate")
                print(f"Progress saved at row {idx}/{total_rows}")
                
    except Exception as e:
        print(f"Error during processing: {e}")
        save_progress(df_target, "error_backup")
        raise e
    
    return df_target

def extract_scores(df):
    columns = ["scaled_score", "logical_equivalence", "signal_behavior", "edge_case_handling", 
               "code_modularity", "resource_efficiency", "timing_pipeline_depth"]
    rows = df.columns
    new_df = pd.DataFrame(columns=columns, index=rows)
    
    for row in rows:
        for col in columns:
            new_df.loc[row, col] = df[row].apply(lambda x: json.loads(x)[col]).mean()
    
    return new_df

if __name__ == "__main__":
    try:
        output_df = pd.DataFrame()
        df = pd.read_csv("data/benchmarking_nl_to_rtl/final_results.csv")
        
        for col in df.columns[2:]:
            print(f"\nProcessing column: {col}")
            output_df = add_score_column(df, output_df, "ground_truth", col, col)
            
            if should_interrupt:
                break
        
        # Calculate and save final scores if we have data
        if not output_df.empty:
            extracted_scores = extract_scores(output_df)
            print("\nFinal scores:")
            print(extracted_scores)
            
            extracted_scores.to_csv("data/benchmarking_nl_to_rtl/final_results_scores.csv")
            output_df.to_csv("data/benchmarking_nl_to_rtl/complete_results.csv")
            
    except Exception as e:
        print(f"\nCritical error in main execution: {e}")
        if 'output_df' in locals() and not output_df.empty:
            save_progress(output_df, "critical_error_backup")
        raise e