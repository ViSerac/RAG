"""
query.py — Interface de perguntas sobre os documentos indexados.
Requer que indexer.py já tenha sido executado.

Uso: python query.py
"""
import os
import sys
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

DOCS_DIR    = "./docs"
CHROMA_DIR  = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL   = "qwen3"
TOP_K       = 4
SCORE_MIN   = 0.30

def carregar():
    if not os.path.exists(CHROMA_DIR):
        print("Erro: rode indexer.py primeiro.")
        sys.exit(1)

    # Vector store
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    # Chunks para BM25 (recarrega os PDFs sem re-embeddar)
    loader = DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    return vectorstore, chunks

def build_retrievers(vectorstore, chunks):
    dense = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = TOP_K

    return dense, bm25

def hybrid_retrieve(pergunta, dense, bm25):
    bm25_docs  = bm25.invoke(pergunta)
    dense_docs = dense.invoke(pergunta)

    scores  = {}
    doc_map = {}

    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content[:80]
        scores[key]  = scores.get(key, 0) + 1 / (rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(dense_docs):
        key = doc.page_content[:80]
        scores[key]  = scores.get(key, 0) + 1 / (rank + 1)
        doc_map[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked[:TOP_K]]

def rerank(pergunta, docs, llm, top_n=2):
    scored = []
    for doc in docs:
        prompt = f"""Numa escala de 0 a 10, qual a relevância deste trecho para responder a pergunta?
Responda APENAS com um número inteiro.

Pergunta: {pergunta}
Trecho: {doc.page_content[:500]}
Relevância:"""

        resposta = llm.invoke(prompt).content.strip()
        try:
            score = int(''.join(filter(str.isdigit, resposta.split('\n')[-1])))
        except:
            score = 0

        scored.append((score, doc))
        print(f"  Rerank score {score}/10 — página {doc.metadata.get('page', '?')}")

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_n]]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_chain(vectorstore, dense, bm25, llm):
    prompt = PromptTemplate.from_template("""
Você é um assistente especializado. Use APENAS o contexto abaixo para responder.
Se a resposta não estiver no contexto, diga "Não encontrei essa informação nos documentos."
Nunca invente informações.

Contexto:
{context}

Pergunta: {question}

Resposta:""")

    return (
        {
            "context": lambda q: format_docs(hybrid_retrieve(q, dense, bm25)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

def debug_retrieval(pergunta, dense, bm25, llm, chain):
    print("\n" + "="*60)
    print(f"RETRIEVAL — '{pergunta}'")
    print("="*60)

    docs_hybrid = hybrid_retrieve(pergunta, dense, bm25)
    print(f"\n[1] Hybrid retrieval — {len(docs_hybrid)} chunks:")
    for i, doc in enumerate(docs_hybrid):
        pagina = doc.metadata.get("page", "?")
        texto  = doc.page_content[:150].replace("\n", " ")
        print(f"  Chunk {i+1} | Página {pagina}: {texto}...")

    print(f"\n[2] Reranking:")
    docs_reranked = rerank(pergunta, docs_hybrid, llm, top_n=2)

    print(f"\n[3] Chunks finais ({len(docs_reranked)}):")
    for i, doc in enumerate(docs_reranked):
        pagina = doc.metadata.get("page", "?")
        texto  = doc.page_content[:200].replace("\n", " ")
        print(f"  Chunk {i+1} | Página {pagina}: {texto}...")

    print("="*60 + "\n")

    resposta = chain.invoke(pergunta)
    print(f"Resposta: {resposta}\n")

def main():
    print("Carregando...")
    vectorstore, chunks = carregar()
    dense, bm25 = build_retrievers(vectorstore, chunks)
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    chain = build_chain(vectorstore, dense, bm25, llm)

    collection = vectorstore._collection
    print(f"Pronto. {collection.count()} chunks indexados.\n")
    print("Comandos:")
    print("  /debug <pergunta>  — pipeline completo de retrieval")
    print("  /info              — configurações")
    print("  sair               — encerra\n")

    while True:
        entrada = input("Pergunta: ").strip()
        if not entrada:
            continue

        if entrada.lower() == "sair":
            break

        elif entrada.startswith("/debug "):
            pergunta = entrada[7:]
            debug_retrieval(pergunta, dense, bm25, llm, chain)

        elif entrada == "/info":
            print(f"\nEmbedding: {EMBED_MODEL}")
            print(f"LLM:       {LLM_MODEL}")
            print(f"Top-K:     {TOP_K}")
            print(f"Score min: {SCORE_MIN}")
            print(f"Retriever: Hybrid BM25 + Dense (RRF)")
            print(f"Reranker:  LLM-based (top 2)\n")

        else:
            resultados = vectorstore.similarity_search_with_score(entrada, k=1)
            melhor = 1 - resultados[0][1]
            if melhor < SCORE_MIN:
                print(f"\n  Score {melhor:.3f} — fora do escopo.\n")
            else:
                resposta = chain.invoke(entrada)
                print(f"\nResposta: {resposta}\n")

if __name__ == "__main__":
    main()