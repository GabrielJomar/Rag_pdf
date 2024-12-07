import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

documents = [
    {"path": "D:\\Applications\\Alameno\\Backend(LLM)-Assignment\\tsla-20231231-gen.pdf", "tag": "Tesla"},
    {"path": "D:\\Applications\\Alameno\\Backend(LLM)-Assignment\\uber-10-k-2023.pdf", "tag": "Uber"},
    {"path": "D:\\Applications\\Alameno\\Backend(LLM)-Assignment\\goog-10-k-2023 (1).pdf", "tag": "Google"}
]

# Initialize an empty list to hold all chunks.Load and process each document, Split and chunk the data and add metadata to each tag.
all_chunks = []
for doc in documents:
    loader = UnstructuredPDFLoader(file_path=doc["path"])
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    
    for chunk in chunks:
        chunk.metadata["source"] = doc["tag"]
    
    all_chunks.extend(chunks)

# Add the chunks to the vector database with embeddings
vector_db = Chroma.from_documents(
    documents=all_chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
    collection_name="local-rag",
    persist_directory="./chroma_db"  
)

# Initialize LLM and retriever
local_llm = "llama3.1"
llm = ChatOllama(model=local_llm)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI Language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines.
    Original question: {question} """
)

def determine_sources(query):
    keywords_to_tags = {
        "tesla": "Tesla",
        "uber": "Uber",
        "google": "Google"
    }
    relevant_tags = [
        tag for keyword, tag in keywords_to_tags.items() if keyword.lower() in query.lower()
    ]
    return relevant_tags

# Custom filtering retriever
def filtered_retriever(query):
    relevant_sources = determine_sources(query)
    if not relevant_sources:
        return vector_db.as_retriever()  # No filtering, search all chunks
    
    retriever = vector_db.as_retriever(search_kwargs={
        "filter": {"source": {"$in": relevant_sources}}
    })
    return retriever

retriever = MultiQueryRetriever.from_llm(
    retriever=filtered_retriever(""),
    llm=llm,
    prompt=QUERY_PROMPT
)

# RAG Prompt for response generation
template = """Answer the question based ONLY on the following context: 
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Chain definition
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

st.title("Document Query Interface")
query = st.text_input("Enter your query:")

if query:
    retriever.retriever = filtered_retriever(query)
    response = chain.invoke(query)

    st.subheader("Response")
    st.write(response)

