from flask import Flask, request, jsonify
from pymongo import MongoClient
import gradio as gr

app = Flask(__name__)
client = MongoClient('mongodb://localhost:27017/')
db = client['question_answer_db']
collection = db['questions_answers']

def QAModel(query):
    # Implement your QAModel logic here
    pass

def query_data(query):
    # Implement your query_data logic here
    pass

def store_results(question, model_output, model_name, as_output, retriever_output):
    result = collection.insert_one({
        'question': question,
        'answer': model_output['answer'],
        'performance_metrics': {
            'output1': as_output,
            'output2': retriever_output,
            'output3': model_output['answer'],
            'output4': model_output['score'],
            'output5': model_name,
            'output6': model_output['end'],
            'output7': model_output['start']
        }
    })
    return str(result.inserted_id)

def button_click(query):
    model_output, model_name = QAModel(query)
    as_output, retriever_output = query_data(query)
    store_results(query, model_output, model_name, as_output, retriever_output)
    return model_output['answer']

def gradio_interface():
    with gr.Blocks(theme=gr.theme.Base(), title="Question Answering App using Vector Search + Rag") as demo:
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

        button.click(button_click, inputs=[textbox], outputs=[output1, output2, output3, output4, output5, output6, output7])

    return demo

@app.route('/submit_question', methods=['POST'])
def submit_question():
    question = request.json.get('question')
    model_output, model_name = QAModel(question)
    as_output, retriever_output = query_data(question)
    result_id = store_results(question, model_output, model_name, as_output, retriever_output)
    return jsonify({'message': 'Question answered and performance metrics stored in MongoDB!', 'id': result_id})

@app.route('/')
def home():
    return gradio_interface()()

if __name__ == '__main__':
    app.run(debug=True)
