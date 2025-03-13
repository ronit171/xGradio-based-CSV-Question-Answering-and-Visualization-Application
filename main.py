import gradio as gr
import pandas as pd
import json
import matplotlib.pyplot as plt
from pydantic import BaseModel
import ollama

# Define the AI model using Pydantic and Ollama
class CSVQueryModel(BaseModel):
    query: str
    
    def run(self, df: pd.DataFrame):
        try:
            response = ollama.chat(model="llama3", messages=[{"role": "user", "content": self.query}])
            return response["message"]["content"]
        except Exception as e:
            return f"AI Model Error: {str(e)}. Please ensure Ollama is running."

# Function to handle CSV upload
def load_csv(file):
    try:
        df = pd.read_csv(file.name, encoding='latin1', delimiter=',')
        return df, "CSV loaded successfully!"
    except Exception as e:
        return None, f"Error loading CSV: {str(e)}"

# Function to process queries
def query_csv(file, query):
    df, msg = load_csv(file)
    if df is None:
        return msg
    model = CSVQueryModel(query=query)
    return model.run(df)

# Function to generate a graph
def plot_graph(file, x_col, y_col, graph_type):
    df, msg = load_csv(file)
    if df is None:
        return msg
    try:
        plt.figure()
        if graph_type == "Bar Chart":
            df.plot(kind="bar", x=x_col, y=y_col)
        elif graph_type == "Line Chart":
            df.plot(kind="line", x=x_col, y=y_col)
        elif graph_type == "Scatter Plot":
            df.plot(kind="scatter", x=x_col, y=y_col)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"{graph_type} of {y_col} vs {x_col}")
        plt.savefig("plot.png")
        return "plot.png"
    except Exception as e:
        return f"Error generating graph: {str(e)}"

# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# CSV Question Answering & Visualization")
    file = gr.File(label="Upload CSV")
    
    with gr.Row():
        query = gr.Textbox(label="Enter your question")
        query_btn = gr.Button("Ask")
        response = gr.Textbox(label="Response")
        query_btn.click(query_csv, inputs=[file, query], outputs=response)
    
    with gr.Row():
        x_col = gr.Textbox(label="X-axis Column")
        y_col = gr.Textbox(label="Y-axis Column")
        graph_type = gr.Radio(["Bar Chart", "Line Chart", "Scatter Plot"], label="Graph Type")
        graph_btn = gr.Button("Generate Graph")
        graph_output = gr.Image()
        graph_btn.click(plot_graph, inputs=[file, x_col, y_col, graph_type], outputs=graph_output)
    
    gr.Markdown("Upload a CSV, ask questions, and generate visualizations!")

demo.launch(share=True)
