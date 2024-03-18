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