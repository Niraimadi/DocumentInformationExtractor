from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
import os

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def document_assistant(folderpath, query):
    files = os.listdir(folderpath)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(files)
    all_docs = []
    for file in files:
        filepath = os.path.join(f"files/{file}")
        loader = PyPDFLoader(filepath)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        docs = text_splitter.split_documents(documents=documents)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        all_docs.extend(docs)
    print(all_docs)
    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs
    )

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(all_docs, embeddings)

    # Save and reload the vector store
    vectorstore.save_local("faiss_index_")
    persisted_vectorstore = FAISS.load_local("faiss_index_", embeddings, allow_dangerous_deserialization=True)

    llm = OllamaLLM(model="llama3.2")

    template = """
    You are a document assistant. Given the context of the document, you will provide answers to the query based on the information provided in the document.
    Query:
    {query}

    Document Content:
    {context}

    Please do the following:
    - Extract the relevant information from the document context for the given query.
    - Generate an appropriate answer for the qurey based on the information retrieved.
    - Provide the references from the document for the answer generated.
    
    Answer in the following format:
    Query: <query>
    Answer:
    <answer for the query>
    References:
    <references from the document>
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Create a retriever
    qa_chain = (
        {
            "context": persisted_vectorstore.as_retriever() | format_docs,
            "query": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    response = qa_chain.invoke(query)
    return response