from key import MONGOURI
from Rules import NFLRule, parse_rules 
import pymongo
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.chains import RetrievalQA
from Model.QASystemModel import QAModel
from gensim.models import Word2Vec
import os
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from nltk.tokenize import sent_tokenize
import numpy as np
from key import openAI

model=QAModel()
#content=NFLRule()
#model_name, tokenizer, model=QAModel()

#parserules = parse_rules(content)



client = pymongo.MongoClient(MONGOURI)
#db= client.QA_System
#collection = db.NFL_Rules

#results = collection.insert_many(parserules)
#print(f"Inserted {len(results.inserted_ids)} documents")

dbName='QA_System'
collectionName="Results"
collection=client[dbName][collectionName]
path ="/Users/tutudaranijo/Downloads/Github_projects/Python_Project/FootballQASystem/sample_files"
loader = DirectoryLoader(path, glob = "./*.txt", show_progress=True)
data = loader.load()

embeddings=OpenAIEmbeddings(openai_api_key=openAI)

# Now create a vector index
vector_search = MongoDBAtlasVectorSearch.from_documents(data, embedding=embeddings, collection=collection)

def query_data(query):
    docs = vector_search.similarity_search(query, k=1)
    as_output=docs[0].page_content
    llm= model
    retriever =vector_search.as_retriever()
    qa=RetrievalQA.from_chain_type(llm,chain_type="stuff", retriever=retriever)
    retriever_output = qa.run(query)

    return as_output, retriever_output
