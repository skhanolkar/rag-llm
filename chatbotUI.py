import streamlit as st


from llama_index import VectorStoreIndex
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
import chromadb


from ast import Str
import logging
from pydoc import doc
from re import S
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


from llama_index.llms import HuggingFaceLLM
from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
import os
os.environ['NUMEXPR_MAX_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
import numexpr as ne

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
#from langchain import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
import json
# Using LlamaIndex as a Callable Tool
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import VectorStoreIndex,SimpleDirectoryReader,StorageContext,ServiceContext
from llama_index.vector_stores import ChromaVectorStore


import torch

from typing import Optional, List, Mapping, Any


from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt




from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    SummaryIndex
)
from llama_index.callbacks import CallbackManager
from llama_index.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.llms.base import llm_completion_callback



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
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    embed_model = LangchainEmbedding(embeddings)

    return embed_model


def get_llama_service_context(llm,embed_model):
        #Create servicecontext with all local - embedding and llm
    service_context = ServiceContext.from_defaults(
        chunk_size=256,
        llm=llm,
        embed_model=embed_model
    )

    return service_context

def get_llm():

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
        # set to at least 1 to use GPU - add -1
        model_kwargs={"n_gpu_layers": 0},
        # transform inputs into Llama2 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )

    return llm
    


def get_my_llama_index():
    #embed_model = get_embedding_model()
    embed_model=get_embedding_model_from_local()
    print(f"The embed_model is {embed_model}")

    llm = get_llm()
    service_context = get_llama_service_context(llm=llm,embed_model=embed_model)

    db = chromadb.PersistentClient(path="./chroma_db")
    print(f"db  is {db}")
    chroma_collection = db.get_or_create_collection("quickstart")
    print(f"chroma_collection  is {chroma_collection}")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        service_context=service_context,
    )

    return index

index = get_my_llama_index()
chat_engine = index.as_chat_engine(chat_mode="condense_question")

st.title("Chat with your documents!")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

def generate_response(prompt_input):
    return chat_engine.chat(prompt_input).response  


if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking ... "):
            response = generate_response(prompt)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)