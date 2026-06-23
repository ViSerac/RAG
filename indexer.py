"""
indexer.py — Indexa os PDFs da pasta /docs no vector store.
Execute sempre que adicionar novos documentos.

Uso: python indexer.py
"""
import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DOCS_DIR      = "./docs"
CHROMA_DIR    = "./chroma_db"
EMBED_MODEL   = "nomic-embed-text"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

def indexar():
    if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
        print(f"Erro: nenhum PDF encontrado em {DOCS_DIR}/")
        return

    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print("Índice anterior removido.")

    loader = DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    print(f"Carregados: {len(docs)} páginas")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Gerados: {len(chunks)} chunks")

    print("Gerando embeddings (pode demorar)...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print(f"\nPronto. {len(chunks)} chunks indexados em {CHROMA_DIR}/")
    print("Rode: python query.py")

if __name__ == "__main__":
    indexar()