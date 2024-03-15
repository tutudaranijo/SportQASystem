import gradio as gr
from gradio.themes.base import Base
from Backend.Main import query_data
from pydantic import BaseModel
from langchain.chains import RetrievalQA
#def question_answering_app():


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

    button.click(query_data, textbox, outputs=[output1, output2])

demo.launch()

   


