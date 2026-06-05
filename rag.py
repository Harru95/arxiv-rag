import os
import fitz
import requests
import tempfile
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from typing import Generator

load_dotenv()

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm=ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

def download_and_parse_pdf(url:str)->list[Document]:
    if "arxiv.org/abs/" in url:
        url=url.replace("/abs/","/pdf/")

    response=requests.get(url,timeout=30)
    if response.status_code !=200:
        raise ValueError(f"Could not download PDF from {url}")
    
    with tempfile.NamedTemporaryFile(suffix=".pdf",delete=False) as tmp:
        tmp.write(response.content)
        tmp_path=tmp.name

    documents=[]
    pdf=fitz.open(tmp_path)
    for page_num in range(len(pdf)):
        page=pdf[page_num]
        text=page.get_text()
        if text.strip():
            documents.append(Document(
                page_content=text,metadata={"source":url,"page":page_num+1}
            ))
    pdf.close()
    os.unlink(tmp_path)
    return documents

def chunk_document(documents:list[Document])->list[Document]:
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n","\n","."," "]
    )
    return splitter.split_documents(documents)

def build_index(urls:list[str])-> FAISS:
    all_chunks=[]

    for url in urls:
        url=url.strip()
        if not url:
            continue
        documents=download_and_parse_pdf(url)
        chunks=chunk_document(documents)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("No content from the provided URLs")
    
    index=FAISS.from_documents(all_chunks,embeddings)
    return index

def save_index(index:FAISS,path:str="faiss_store"):
    index.save_local(path)
    print(f"Index saved to {path}/")

def load_index(path: str = "faiss_store") -> FAISS:
    if not os.path.exists(path):
        return None
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def query(index:FAISS,question:str,k:int=5)->dict:
    retriever=index.as_retriever(search_kwargs={"k":k})
    relevant_chunks=retriever.invoke(question)

    context="\n\n".join([
        f"[Source:{doc.metadata['source']},Page {doc.metadata['page']}]\n{doc.page_content}"
        for doc in relevant_chunks
    ])

    prompt=f"""You are a research assistant.Answer the question based ONLY on the context provided below.
For every claim you make, mention the page number it came from.
If the answer is not in the context,say "I could not find this in the provided papers."

Context:
{context}

Question:{question}

Answer:"""
    
    response=llm.invoke(prompt)

    sources=list(set([
        f"{doc.metadata['source']} - Page {doc.metadata['page']}"
        for doc in relevant_chunks
    ]))

    return{
        "answer": response.content,
        "sources":sources
    }


def query_stream(index:FAISS,question:str,k:int=5)-> Generator:
    retriever=index.as_retriever(search_kwargs={"k":k})
    relevant_chunks=retriever.invoke(question)

    context="\n\n".join([
        f"[Source:{doc.metadata['source']},Page {doc.metadata['page']}]\n{doc.page_content}"
        for doc in relevant_chunks
    ])

    prompt = f"""You are a research assistant. Answer the question based ONLY on the context provided below.
For every claim you make, mention the page number it came from.
If the answer is not in the context, say "I could not find this in the provided papers."

Context:
{context}

Question:{question}
Answer:"""
    
    for chunk in llm.stream(prompt):
        if chunk.content:
            yield chunk.content

    sources=list(set([
        f"{doc.metadata['source']} - Page {doc.metadata['page']}"
        for doc in relevant_chunks
    ]))
    import json
    yield f"\n__SOURCES__{json.dumps(sources)}"