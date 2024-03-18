import gradio as gr
from gradio.themes.base import Base
from pydantic import BaseModel
from langchain.chains import RetrievalQA
#from Backend.Main import vector_search
from Model.QASystemModel import QAModel
model=QAModel()
model_name, tokenizer, model=QAModel()

def query_data(query):
    docs = vector_search.similarity_search(query, k=1)
    as_output=docs[0].page_content
    llm= model
    retriever =vector_search.as_retriever()
    qa=RetrievalQA.from_chain_type(llm,chain_type="stuff", retriever=retriever)
    retriever_output = qa.run(query)

    return as_output, retriever_output

def website(querydata):
    with gr.Blocks(theme=Base(), title="Question Answering App using Vector Search + Rag") as demo:
        gr.Markdown(
            """
            # Question Answering App using Atlas Vector Search + Rag Architeture
            """
        )
        textbox = gr.Textbox(label="Enter your Question")

        with gr.Row():
            button = gr.Button("Submit", variant="primary")

        with gr.Column():
            output1 = gr.Textbox(lines=1, max_lines=20, label="Output with Atlas Vector Search(return as is)")
            output2 = gr.Textbox(lines=1, max_lines=20, label="Output with chaining Atlas to langchain retrieval")

        button.click(querydata, textbox, outputs=[output1, output2])

    demo.launch()

   


