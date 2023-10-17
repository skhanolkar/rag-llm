# rag-llm
rag+llm
This repo implements RAG using :
1. BAAI Embeddings(BAAI/bge-large-en-v1.5) - Locally deployed on ec2. Embedding size is 1024
2. Mistral: Open source model :https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF - This is quantized model.
3. llama-index: For creating document index and act as data orchestration framework.
4. Chroma-dd as vector store.
5. Streamlit for Chatbot interface.

Data used :
1. White paper - Attention is all you need.
2. Home Exercise recommendation after Hip replacement
3. Amazon 2022 financial report.


Look and feel:
![plot](./images/Chat-with-document.png)