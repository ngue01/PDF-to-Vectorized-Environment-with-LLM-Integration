import time
import chromadb
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings #sbert
#from langchain.embeddings import SpacyEmbeddings
#from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
persist_directory = "./data"
sbert_embedding = "all-MiniLM-L6-v2"# slower "all-mpnet-base-v2"

"""to store pdf to vectorized environment """

def StorePDF(client, col_name,pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
    texts = text_splitter.split_documents (documents)
    embeddings =SentenceTransformerEmbeddings(
        model_name=sbert_embedding, 
        ) #SpacyEmbeddings() #OpenAIEmbeddings()

    vectordb = Chroma.from_documents (documents=texts,
            embedding=embeddings, 
            persist_directory=persist_directory,
            collection_name=col_name,
            client=client)
    return vectordb

def GetVectorDB(client, collection,pdf):
    vectordb = None
    cols= [col.name==collection for col in client.list_collections()]
    if len(cols)<1:
        ## first run only
        vectordb = StorePDF (client=client,col_name=collection,pdf_path=pdf)
    else:
        ##when db already exists
        vectordb = Chroma (client=client,
            persist_directory=persist_directory,
            collection_name=collection,
            embedding_function=SentenceTransformerEmbeddings (model_name=sbert_embedding))
    return vectordb

client = chromadb.PersistentClient(path=persist_directory)

retriever = GetVectorDB(client, "acu","./data/acu.pdf").as_retriever(search_kwargs={"k": 3})

llm =LlamaCpp(#https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML
        model_path="models/llama2_ggml/llama-2-7b-chat.ggmlv3.q4_0.bin",
        #verbose=True,
        max_tokens=128,
        n_gpu_layers=12,
        n_ctx=2048,
)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, )

while True:
    user_input = input("Enter a query: ")
    if user_input == "exit":
        break
    start_time = time.time()
    query = f"###Prompt {user_input}"
    try:
        llm_response = qa (query)
        print(llm_response["result"],f" (exec time : {time.time()-start_time:.2f}s)") 
    except Exception as err:
        print('Exception occurred. Please try again', str(err))