import os
from together import Together
import json
import numpy as np
import random
from pydantic import BaseModel, Field
from openai import OpenAI
import faiss
import pandas as pd

TOGETHER_API_KEY=os.environ.get("TOGETHER_API_KEY")
RUNPOD_API_KEY=os.environ.get("RUNPOD_API_KEY")
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")

together_client = Together(api_key=TOGETHER_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
runpod_client = OpenAI(
  api_key=RUNPOD_API_KEY, 
  base_url="https://api.runpod.ai/v2/h9yzodruyoypx7/openai/v1" # TODO: Change this to your endpoint URL
)

model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" #"meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
openai_model_name = "gpt-4o"
debug = True
rerank = True
k = 100 # number of documents to retrieve 
n = 20 # number of documents to rerank
use_openai = False

print("Welcome\n_______________________\nConfig:")
print(f"LLaMa Model: {model_name}")
print(f"OpenAI Model: {openai_model_name}")
print(f"Debug Mode: {debug}")
print(f"Rerank Mode: {rerank}")
print(f"top-k (Used in KNN search): {k}")
print(f"top-n (Used in Re-ranking): {n}")
print(f"Use OpenAI: {use_openai}")
print("_______________________\n")

print("Loading index")
my_index = faiss.read_index("knn.index")
print("Loading texts")
texts = pd.read_json("texts.json", lines=True).text.tolist()
print(f"Done loading {len(texts)} texts\n_______________________\n")

def generate_response(
    response_format=None,
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    messages=[
      {"role": "system", "content": "You are a helpful assistant that answers a user's questions based on the information available to you."},
      {"role": "user", "content": "What are some fun things to do in Delhi?"}], 
    print=False,
):
    if print:
        print(f"Generating response for {model}")
    if response_format:
      response = together_client.chat.completions.create(
          model=model,
          messages=messages,
          response_format=response_format
      )
    else:
      response = together_client.chat.completions.create(
          model=model,
          messages=messages
      )
    return response.choices[0].message.content

def generate_response_openai(
    messages,
    model="gpt-4o",
    response_format=None,
):
    model=openai_model_name
    if response_format:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"}
        )
    else:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages
        )
    return response.choices[0].message.content

def create_init_prompt(conversation_history):
    SYSTEM_PROMPT=open("SYSTEM_PROMPT_query_generator.txt", "r").read()
    USER_PROMPT=f"Below is the conversation history between a user and an assistant. You should create queries based on the user's needs. You should return the result in json format.\nConversation History:\n{conversation_history}"
    return SYSTEM_PROMPT, USER_PROMPT

def create_answer_prompt(query_results, user_message):
    # Query results is a list of tuples (result, score)
    SYSTEM_PROMPT=open("SYSTEM_PROMPT_assistant.txt", "r").read()
    if len(query_results) > 0:
        USER_PROMPT=f"{user_message}\n\nHere are the results for the queries created with the user's message.\nRETRIEVED CONTEXT:\n{query_results}"
    else:
        USER_PROMPT=f"{user_message}"
    return SYSTEM_PROMPT, USER_PROMPT

def create_embeddings(queries = ["gutenberg project created by whom?"]):
    embeddings = runpod_client.embeddings.create(
        model="BAAI/bge-base-en-v1.5",
        input=queries
    )
    query_vectors = []
    for embedding in embeddings.data:
        query_vectors.append(embedding.embedding)
    return np.array(query_vectors)  # This will return a 2D array with shape (num_queries, 768)

def query_index(queries, k=50, debug=False):
    query_vectors = create_embeddings(queries)
    distances, indices = my_index.search(query_vectors, k)
    top_texts = dict()

    if debug:
        print(f"Retrieved {k*len(queries)} results for {len(queries)} queries")
        
    for i, vector in enumerate(query_vectors):
        d = distances[i]
        idx = indices[i]
        top_texts[i] = {
            "query": queries[i],
            "top_texts": [texts[j] for j in idx]
        }
    return top_texts

def get_query_doc_pairs(query_results):
    query_doc_pairs = []
    for item in query_results.values():
        query_doc_pairs.append((item["query"], item["top_texts"]))
    return query_doc_pairs

def rerank_query_doc_pairs(old_query_doc_pairs, top_n=5):
    new_query_doc_pairs = []
    for query, docs in old_query_doc_pairs:
        response = together_client.rerank.create(
            model="Salesforce/Llama-Rank-V1",
            query=query,
            documents=docs,
            top_n=top_n
        )
        top_docs = [docs[i.index] for i in response.results]
        new_query_doc_pairs.append((query, top_docs))
    return new_query_doc_pairs

class QueryResults(BaseModel):
    query_results: list[str] = Field(description="A list of queries in a json format")

response_format_query = {"type": "json_object", "schema": QueryResults.model_json_schema()}

def chat_loop(debug=False, rerank=False, k=25, n=5):
    """
    Chat loop that takes in user input and generates a response.
    First runs the query generator to create queries (can be empty).
    Then runs the retriever to get the top k documents for each query.
    Then runs the re-ranker to re-rank the documents for each query.
    Then runs the response generator to generate a response to the user's question.
    """
    chat_history = []
    while True:
        print("_______________________")
        user_message = input("You: ")
        if user_message == "exit":
            break
        msg = {"role": "user", "content": user_message}
        chat_history.append(msg)
        
        system_prompt, user_prompt = create_init_prompt(chat_history)
        if debug:
            print(user_prompt)
        if use_openai:
            response = generate_response_openai(response_format=response_format_query, model=model_name, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
        else:
            response = generate_response(response_format=response_format_query, model=model_name, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
        if debug:
            print(response)
        
        queries = json.loads(response)['query_results']
        if len(queries) > 0:
            if debug:
                print(queries)
            query_results = query_index(queries, k=k, debug=debug)
            query_doc_pairs = get_query_doc_pairs(query_results)
        else:
            if debug:
                print("\nNot creating queries")
            query_doc_pairs = []
        
        if rerank:
            query_doc_pairs = rerank_query_doc_pairs(query_doc_pairs, top_n=n)
        
        if debug:
            print("\nCreating answer prompt")
        system_prompt, user_prompt = create_answer_prompt(query_doc_pairs, user_message)
        if debug:
            print(user_prompt)
        if use_openai:
            response = generate_response_openai(response_format=None, model=model_name, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
        else:
            response = generate_response(response_format=None, model=model_name, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
        print(f"\nAssistant: {response}\n")
        
        chat_history.pop()
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": response})
    

if __name__ == "__main__":
    print("Starting chat")
    print("Use 'exit' to end the chat")
    chat_loop(debug=debug, rerank=rerank, k=k, n=n)
