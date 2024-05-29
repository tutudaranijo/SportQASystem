
import os

MONGOURI = os.environ.get('MONGOURI')
openAI = os.environ.get('openAI')
hugembed = os.environ.get('hugembed')
from Backend.Rules import NFLRule, parse_rules 
import pymongo
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import requests
from langchain.retrievers import ParentDocumentRetriever
import pypdf
from langchain_community.document_loaders import PyPDFLoaders

class DocumentWrapper:
    def __init__(self, content, metadata = None):
        self.page_content = content
        self.metadata = metadata if metadata is not None else {}
# Function to check if the collection exists and has documents
def check_collection(client :str, db_name:str, collection_name:str):
    db = client[db_name]
    if collection_name in db.list_collection_names():
        collection = db[collection_name]
        count = collection.count_documents({})
        if count > 0:
            print(f"Collection '{collection_name}' exists and has {count} documents.")
            return True
        else:
            print(f"Collection '{collection_name}' exists but is empty.")
    else:
        print(f"Collection '{collection_name}' does not exist.")
    return False

# Initialize MongoDB client
client = pymongo.MongoClient(MONGOURI)
DB_NAME = "langchain_db"
COLLECTION_NAME = "test"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "index_name"

MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]
loader = PyPDFLoader("https://arxiv.org/pdf/2303.08774.pdf")
data = loader.load()
embedding=OpenAIEmbeddings(disallowed_special=(), api_key=openAI)
query_sentence= "What were the compute requirements for training GPT 4"
# Check if NFLRules collection has documents
if check_collection(client, 'langchain_db', 'test'):
    # If the test collection exists, run the vector search
    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        MONGOURI,
        DB_NAME + "." + COLLECTION_NAME,
        OpenAIEmbeddings(disallowed_special=(), api_key=openAI),
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )
#    db = client['langchain_db']
#    collection = db['test']
#    query = query_sentence
#
#    vectorstore = MongoDBAtlasVectorSearch(collection, embedding)
#
#    docs = vectorstore.similarity_search(query, K=1)
#    as_ouput= docs[0].page_content
#    print(as_ouput)

else:
    # If the test collection does not exist, load, parse, and insert the NFL rules
    content = NFLRule()
    documents = parse_rules(content)
    page_contents = [DocumentWrapper(doc['page_content']) for doc in documents]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(data)

    # Insert the documents in MongoDB Atlas with their embedding
    vector_search = MongoDBAtlasVectorSearch.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(disallowed_special=(), api_key=openAI),
        collection=MONGODB_COLLECTION,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )


    
 #   db = client['langchain_db']
 #   collection = db['test']
 #   query = query_sentence
#
 #   vectorstore = MongoDBAtlasVectorSearch(collection, embedding)
#
 #   docs = vectorstore.similarity_search(query, K=1)
 #   as_ouput= docs[0].page_content
 #   print(as_ouput)
#    results = collection.aggregate([
#  {
#    "$vectorSearch": {
#      "index": "vector_index",
#      "path": "embedding",
#      "queryVector": embedding(query),
#      "numCandidates": 100,
#      "limit": 4
#    }
#  }
#])
#    for document in results:
#        print(document) 
#
