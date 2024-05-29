import gradio as gr
from gradio.themes.base import Base
from langchain.chains import RetrievalQA
import pymongo
from langchain_mongodb import MongoDBAtlasVectorSearch
import sys
import os
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_directory)
from Backend.key import MONGOURI, hugembed
from Model.QASystemModel import QAModel
from Backend.Rules import generate_embedding
from Backend.Main import data
from Backend.key import openAI
from transformers import DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer, DPRReaderTokenizer, DPRQuestionEncoder
embedding=OpenAIEmbeddings(disallowed_special=(), api_key=openAI)
client = pymongo.MongoClient(MONGOURI)
dbName='langchain_db'
collectionName="SportsRules"
collection=client[dbName][collectionName]

vector_search= MongoDBAtlasVectorSearch(collection,embedding)
def query_data(query):
    try:
        docs = vector_search.similarity_search(query, k=1)
        if docs:  
            as_output = docs[0].page_content   
            llm=OpenAI(api_key=openAI, temperature=0)
            retriever = vector_search.as_retriever()
            qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
            retriever_output = qa.run(query)
            
        else:
            return "No documents found", None
    except Exception as e:
        return str(e), None

    return as_output, retriever_output

def website():
    with gr.Blocks(theme=Base(), title="Question Answering App using Vector Search + Rag") as demo:
        gr.Markdown(
            """
            # Question Answering App using Atlas Vector Search + Rag Architecture
            """
        )
        textbox = gr.Textbox(label="Enter your Question")

        with gr.Row():
            button = gr.Button("Submit", variant="primary")

        with gr.Column():
            output1 = gr.Textbox(lines=1, max_lines=20, label="Output with Atlas Vector Search(return as is)")
            output2 = gr.Textbox(lines=1, max_lines=20, label="Output with chaining Atlas to langchain retrieval")
            output3 = gr.Textbox(lines=1, max_lines=20, label="Answer")
            output4 = gr.Textbox(lines=1, max_lines=20, label="Score")
            output5 = gr.Textbox(lines=1, max_lines=20, label="Model")
            output6 = gr.Textbox(lines=1, max_lines=20, label="end")
            output7 = gr.Textbox(lines=1, max_lines=20, label="start")

        def button_click(query):
            model_output, model_name = QAModel(query)
            as_output, retriever_output = query_data(query)
            
            # Initialize output values
            output_values = ["", "", "", "", "", "", ""]
            
            # Set output values if data is available
            if as_output:
                output_values[0] = as_output
            if retriever_output:
                output_values[1] = retriever_output
            if model_output:
                output_values[2] = model_output['answer']
                output_values[3] = model_output['score']
                output_values[4] = model_name
                output_values[5] = model_output['end']
                output_values[6] = model_output['start']

            # Return output values
            return output_values

        button.click(button_click, inputs=[textbox], outputs=[output1, output2, output3, output4, output5, output6, output7])

    demo.launch()

website()


