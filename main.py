import os
import sys
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_DIR  = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL   = "qwen3"
TOP_K       = 4
SCORE_MIN   = 0.30

# ─── 1. LOAD ────────────────────────────────────────────────────
loader = PyPDFLoader("documento.pdf")
docs = loader.load()
print(f"Carregados {len(docs)} páginas")

# ─── 2. SPLIT ───────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(docs)
print(f"Gerados {len(chunks)} chunks")

# ─── 3. EMBED + ARMAZENAR ───────────────────────────────────────
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
)
print("Vector store criado\n")

# ─── 4. RETRIEVERS ──────────────────────────────────────────────
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = TOP_K

def hybrid_retrieve(pergunta: str, k: int = TOP_K) -> list:
    """Combina BM25 + Dense manualmente com reciprocal rank fusion."""
    bm25_docs  = bm25_retriever.invoke(pergunta)
    dense_docs = dense_retriever.invoke(pergunta)

    scores = {}
    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0) + 1 / (rank + 1)
        scores[key + "_doc"] = doc

    for rank, doc in enumerate(dense_docs):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0) + 1 / (rank + 1)
        scores[key + "_doc"] = doc

    ranked = sorted(
        [(k, v) for k, v in scores.items() if not k.endswith("_doc")],
        key=lambda x: x[1],
        reverse=True,
    )

    return [scores[key + "_doc"] for key, _ in ranked[:k]]

# ─── 5. RERANKER ────────────────────────────────────────────────
llm = ChatOllama(model=LLM_MODEL, temperature=0)

def rerank(pergunta: str, docs: list, top_n: int = 2) -> list:
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

# ─── 6. PROMPT ──────────────────────────────────────────────────
prompt = PromptTemplate.from_template("""
Você é um assistente especializado. Use APENAS o contexto abaixo para responder.
Se a resposta não estiver no contexto, diga "Não encontrei essa informação nos documentos."
Nunca invente informações.

Contexto:
{context}

Pergunta: {question}

Resposta:""")

# ─── 7. CHAIN ───────────────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": lambda q: format_docs(hybrid_retrieve(q)), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ─── 8. DEBUG ───────────────────────────────────────────────────
def debug_retrieval(pergunta):
    print("\n" + "="*60)
    print(f"RETRIEVAL — '{pergunta}'")
    print("="*60)

    docs_hybrid = hybrid_retrieve(pergunta)
    print(f"\n[1] Hybrid retrieval — {len(docs_hybrid)} chunks:")
    for i, doc in enumerate(docs_hybrid):
        pagina = doc.metadata.get("page", "?")
        texto  = doc.page_content[:150].replace("\n", " ")
        print(f"  Chunk {i+1} | Página {pagina}: {texto}...")

    print(f"\n[2] Reranking:")
    docs_reranked = rerank(pergunta, docs_hybrid, top_n=2)

    print(f"\n[3] Chunks finais após rerank ({len(docs_reranked)}):")
    for i, doc in enumerate(docs_reranked):
        pagina = doc.metadata.get("page", "?")
        texto  = doc.page_content[:200].replace("\n", " ")
        print(f"  Chunk {i+1} | Página {pagina}: {texto}...")

    print("="*60 + "\n")
    return docs_reranked

def responder(pergunta):
    resultados = vectorstore.similarity_search_with_score(pergunta, k=1)
    melhor_score = 1 - resultados[0][1]

    if melhor_score < SCORE_MIN:
        print(f"\n⚠️  Score: {melhor_score:.3f} — fora do escopo dos documentos.\n")
        return

    resposta = chain.invoke(pergunta)
    print(f"\nResposta: {resposta}\n")

# ─── 9. LOOP ────────────────────────────────────────────────────
print("RAG pronto. Comandos:")
print("  /debug <pergunta>  — mostra pipeline completo de retrieval")
print("  /chunks            — mostra os primeiros chunks indexados")
print("  /info              — configurações atuais")
print("  sair               — encerra\n")

while True:
    entrada = input("Pergunta: ").strip()
    if not entrada:
        continue

    if entrada.lower() == "sair":
        break

    elif entrada.startswith("/debug "):
        pergunta = entrada[7:]
        debug_retrieval(pergunta)
        resposta = chain.invoke(pergunta)
        print(f"Resposta: {resposta}\n")

    elif entrada == "/chunks":
        print(f"\nTotal de chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:5]):
            pagina = chunk.metadata.get("page", "?")
            print(f"\nChunk {i+1} (página {pagina}):")
            print(chunk.page_content[:300])
            print("---")
        if len(chunks) > 5:
            print(f"\n... e mais {len(chunks) - 5} chunks\n")

    elif entrada == "/info":
        collection = vectorstore._collection
        print(f"\nVetores indexados: {collection.count()}")
        print(f"Embedding model:   {EMBED_MODEL}")
        print(f"LLM:               {LLM_MODEL}")
        print(f"Top-K:             {TOP_K}")
        print(f"Score mínimo:      {SCORE_MIN}")
        print(f"Retriever:         Hybrid (BM25 + Dense, 50/50)")
        print(f"Reranker:          LLM-based (top 2)\n")

    else:
        responder(entrada)