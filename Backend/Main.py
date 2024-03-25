from key import MONGOURI, openAI, hugembed
from Rules import NFLRule, parse_rules 
import pymongo
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import requests
from langchain.retrievers import ParentDocumentRetriever

# Function to check if the collection exists and has documents
def check_collection(client, db_name, collection_name):
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

# Check if NFLRules collection has documents
if not check_collection(client, 'QA_System', 'NFLRules'):
    # Load and parse the NFL rules
    content = NFLRule()
    parsed_rules = parse_rules(content)

    # Insert parsed rules into the NFLRules collection
    db = client['QA_System']
    collection = db['NFLRules']
    results = collection.insert_many(parsed_rules)
    print(f"Inserted {len(results.inserted_ids)} documents into NFLRules collection.")

# Retrieve documents from NFLRules collection
collection = client['QA_System']['NFLRules']
query = {"Index": {"$exists": True}}  
data = []
try:
    documents = collection.find(query)
    for document in documents:
        data.append(document['page_content'])
    if not data:
        print("No documents found with the specified query.")
except Exception as e:
    print(f"An error occurred while retrieving documents: {e}")

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    api_key=openAI
)

# Initialize MongoDB Atlas Vector Search
namespace = "QA_System.NFLRules"
vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=MONGOURI,
    embedding=embeddings,
    namespace=namespace
)

# Additional components for document retrieval
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter
)

# Add documents to retriever (assuming data has the documents)
retriever.add_documents(documents)

# Perform a test similarity search (adjust as needed)
test_query = "touchdown"  # Replace with a relevant query
sub_docs = vectorstore.similarity_search(query=test_query,k=1)

if sub_docs:
    print(sub_docs[0].page_content)
else:
    print("No similar documents found.")
