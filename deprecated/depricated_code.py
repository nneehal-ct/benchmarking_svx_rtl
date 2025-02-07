# import time
# import weave
# from functools import wraps
# import tiktoken
# from datetime import datetime

# def count_tokens(text: str) -> int:
#     """Helper function to count tokens for a given text"""
#     encoding = tiktoken.encoding_for_model('gpt-4o')
#     return len(encoding.encode(text))

# def retry_decorator(max_retries=3, delay=1):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             retries = 0
#             while retries < max_retries:
#                 try:
#                     return func(*args, **kwargs)
#                 except Exception as e:
#                     retries += 1
#                     if retries == max_retries:
#                         raise e
#                     print(f"Attempt {retries} failed. Retrying in {delay} seconds...")
#                     time.sleep(delay)
#             return None
#         return wrapper
#     return decorator

# # Initialize Weave project
# weave.init('SVx')

# @weave.op()
# def track_token_metrics(messages, response):
#     """Track token usage and other metrics for the LLM call"""
#     # Count input tokens
#     input_tokens = sum(count_tokens(message['content']) for message in messages)
    
#     # Count output tokens
#     output_text = response.choices[0].message.content
#     output_tokens = count_tokens(output_text)
    
#     return {
#         'input_tokens': input_tokens,
#         'output_tokens': output_tokens,
#         'total_tokens': input_tokens + output_tokens,
#     }

# @weave.op()
# @retry_decorator(max_retries=3, delay=1)
# def make_llm_request(client, model, messages):
#     """Make LLM request with automatic input/output tracking"""
#     # Make the API call
#     response = client.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=0.01
#     )
    
#     # Track metrics
#     metrics = track_token_metrics(messages, response)
    
#     return {
#         'response': response.choices[0].message.content,
#         'metrics': metrics,
#         'provider': 'together' if 'together' in str(client) else 'aisuite',
#         'model': model
#     }


#################################


# import aisuite as ai 
# from together import Together 
# import pandas as pd


# def ask_llm(query, query_code):
#     #These are the examples for 2-shot learning, don't use these examples for inference or result benchmarking 
#     example1, example_code1 = get_example(1)
#     example2, example_code2 = get_example(1234)

#     client = ai.Client()

#     models = {
#         "aisuite": {
#             "client": ai.Client(),
#             "models": {
#                 "Qwen-2.5-72B-It": "huggingface:Qwen/Qwen2.5-72B-Instruct",
#                 "Claude-3.5-Sonnet": "anthropic:claude-3-5-sonnet-20240620"
#             }
#         },
#         "together": {
#             "client": Together(),
#             "models": {
#                 "Llama-3.3-70B-It": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
#             }
#         }    
#     }

#     system_message = """You are an helpful assistant. You will be given a query along with two example queries and corresponding answers to follow. \
#         Based on that, you will need to provide a solution to the query. Only answer with Verilog code block, do not generate anything else. Follow the example format for output. \
#         \n\n <Example 1> ##Instruction: {0} ##Query: {1} ##Answer: {2} \n\n <\Example 1> \n\n <Example 2> \
#         ##Instruction: {3} ##Query: {4} ##Answer: {5} \n\n <\Example 2> \n\n <User Query> \n\n"""

#     formatted_system_message = system_message.format(
#         example1['block_summary']['system'],
#         example1['block_summary']['prompt'],
#         example_code1,
#         example2['block_summary']['system'],
#         example2['block_summary']['prompt'],
#         example_code2
#     )

#     user_message = f"{query['block_summary']['system']} {query['block_summary']['prompt']} \n\n </User Query>"

#     messages = [
#         {
#             "role": "system",
#             "content": formatted_system_message
#         },
#         {
#             "role": "user",
#             "content": user_message
#         }
#     ]

#     response_dataframe = pd.DataFrame()

#     # write for loop for traversing each client and model and generate response only for model_names 
#     for client_name, client_data in models.items():
#         print(f"Accessing {client_name} client...")
#         for model_name, model in client_data['models'].items():
#             print(f"Attempting response from {model_name}...")
#             try:
#                 llm_response = make_llm_request(client_data['client'], model, messages)
#                 response_dataframe[f"{client_name}_{model_name}"] = [llm_response['response']] #[response.choices[0].message.content]
#                 print(f"Response from {model_name} generated successfully!")
#             except Exception as e:
#                 print(f"Error after all retries for {model_name}: {e}")
#                 response_dataframe[f"{client_name}_{model_name}"] = [None]
    
#     #response_dataframe['ground_truth'] = query_code
#     #insert in first column 
#     response_dataframe.insert(0, 'ground_truth', query_code)

#     return response_dataframe