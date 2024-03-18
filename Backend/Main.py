from key import MONGOURI, openAI
from Rules import NFLRule, parse_rules 
import pymongo
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain.chains import RetrievalQA
from gensim.models import Word2Vec
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from nltk.tokenize import sent_tokenize
import numpy as np



content=NFLRule()

parserules = parse_rules(content)



client = pymongo.MongoClient(MONGOURI)
db= client.QA_System
collection = db.NFL_Rules

results = collection.insert_many(parserules)
print(f"Inserted {len(results.inserted_ids)} documents")

dbName='QA_System'
collectionName="Results"
collection=client[dbName][collectionName]
path ="/Users/tutudaranijo/Downloads/Github_projects/Python_Project/FootballQASystem/sample_files"
loader = DirectoryLoader(path, glob = "./*.txt", show_progress=True)
data = loader.load()

embeddings=OpenAIEmbeddings(openai_api_key=openAI)

# Now create a vector index
vector_search = MongoDBAtlasVectorSearch.from_documents(data, embedding=embeddings, collection=collection)


