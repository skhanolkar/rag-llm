from ast import Str
import collections
import logging
from pydoc import doc
from re import S
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


import os
os.environ['NUMEXPR_MAX_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
import numexpr as ne

from llama_index import VectorStoreIndex,SimpleDirectoryReader,StorageContext,ServiceContext
from llama_index.vector_stores import ChromaVectorStore
import torch
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.prompts.prompts import SimpleInputPrompt

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext

import chromadb
#Read the system environment
HF_SENTENCE_TRANSFORMER_LOCAL_DIR = os.environ.get('HF_SENTENCE_TRANSFORMER_LOCAL_DIR')
HF_SENTENCE_EMBEDDER_DEVICE= os.environ.get('HF_SENTENCE_EMBEDDER_DEVICE')
LLM_LOCAL_PATH=os.environ.get('LLM_LOCAL_PATH')

def get_embedding_model_from_local():

    print(f"HF_SENTENCE_TRANSFORMER_LOCAL_DIR is {HF_SENTENCE_TRANSFORMER_LOCAL_DIR}")
    model_name = HF_SENTENCE_TRANSFORMER_LOCAL_DIR
    #model_kwargs = {'device': 'cpu'}
    model_kwargs = {'device': HF_SENTENCE_EMBEDDER_DEVICE}
    encode_kwargs = {'normalize_embeddings': False}
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    except Exception as error:
        print("An exception occurred:", error)
    print(f"Here")
    embed_model = LangchainEmbedding(embeddings)
    print(f"Here embed_model is {embed_model}")
    return embed_model



def main(ip_data):
    
    print(f"ip into {ip_data} and {chromadb}")

    db = chromadb.PersistentClient(path="./chroma_db")
    print(f"db  is {db}")
    chroma_collection = db.get_or_create_collection("quickstart")
    print(f"chroma_collection  is {chroma_collection}")
    
    
    embed_model = LangchainEmbedding(
                HuggingFaceEmbeddings(model_name="/home/ec2-user/.cache/torch/sentence_transformers/BAAI_bge-large-en-v1.5")
    )



    documents = SimpleDirectoryReader("./data").load_data()


    system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."

    # This will wrap the default prompts that are internal to llama-index
    query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")
    
    
    # define our LLM
    
    llm = get_huggingface_model()

    #Create servicecontext with all local - embedding and llm
    service_context = ServiceContext.from_defaults(
        chunk_size=256,
        llm=llm,
        embed_model=embed_model
    )

    print(f"service_context is {service_context}")


   
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)

    # Query and print response
    query_engine = index.as_query_engine()
    response = query_engine.query("how to perform Ankle pumps?")
    print(f"The response is {response}")

def get_embedding_model():

    
    embed_model = LangchainEmbedding(
                HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    )

    return embed_model


def get_llama_service_context(llm,embed_model):
        #Create servicecontext with all local - embedding and llm
    service_context = ServiceContext.from_defaults(
        chunk_size=256,
        llm=llm,
        embed_model=embed_model
    )

    return service_context

def get_huggingface_model():

    print(f"loading MIstral ")
    #using TheBlockm - Mistral 
    llm = LlamaCPP(
        # You can pass in the URL to a GGML model to download it automatically
        #model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path=LLM_LOCAL_PATH,
        temperature=0.1,
        max_new_tokens=256,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=3900,
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={"n_gpu_layers": 0},
        # transform inputs into Llama2 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )

    return llm

def load_data_into_chroma_db(ip_data):
    print(f"{ip_data}")
    #define document directory
    documents = SimpleDirectoryReader("./data").load_data()

    #get embedding model 
    #embed_model = get_embedding_model()
    embed_model = get_embedding_model_from_local()
    print(f"The embed_model is {embed_model}")
    llm = get_huggingface_model()
    service_context = get_llama_service_context(llm=llm,embed_model=embed_model)

    db = chromadb.PersistentClient(path="./chroma_db")
    print(f"db  is {db}")
    chroma_collection = db.get_or_create_collection("quickstart")
    print(f"chroma_collection  is {chroma_collection}")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
                                            documents=documents,
                                            storage_context=storage_context,
                                            service_context=service_context)

 

    print(f"index is {index}")



if __name__ =='__main__':
    #main("Test")
    print(f"Test 1")
    load_data_into_chroma_db("Test")