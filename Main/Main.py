# libraries 
from pymongo import MongoClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base
import Key_param 
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from Backend.Rules import NFLRule, remove_special_characters,to_lowercase,remove_stopwords, tokenize_with_nltk,CleanText
from Model.QASystemModel import QAModel


NflRules = NFLRule()
documents=CleanText(NflRules)
