import re
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent


# Reading the Secret Key through the saved file
with open("Secret Key(Helpmate).txt", "r") as file:
    api_k = file.read()

file.close()


# Initialize LLM
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    api_key=api_k,
    frequency_penalty=0.2,
    presence_penalty=0.5,
    streaming=True,
)

# Global variables
agent = None  
df_global = None  

# Function to create an agent after CSV is uploaded
def create_agent(file):
    global agent, df_global  
    try:
        df_global = pd.read_csv(file.name)  
        agent = create_pandas_dataframe_agent(
            model,
            df_global,
            agent_type="tool-calling",
            allow_dangerous_code=True,
            verbose=True,
        )
        return "✅ File uploaded successfully! You can now ask your queries."
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Function to extract only Python code from LLM response
def extract_code(llm_response):
    # Use regex to extract Python code from triple-backtick blocks
    match = re.search(r"```python\n(.*?)```", llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()  # Return only the extracted code
    return llm_response.strip()  # If no backticks, return as-is

# Function to execute the LLM-generated code securely
def execute_plot_code(code):
    global df_global
    try:
        # Ensure Matplotlib figure before execution
        plt.figure(figsize=(8, 5))

        # Define local execution environment
        local_env = {"df": df_global, "sns": sns, "plt": plt}

        # Execute the extracted code
        exec(code, {}, local_env)

        # Save the plot
        plot_path = "generated_plot.png"
        plt.savefig(plot_path)
        plt.close()
        
        return f"✅ Plot generated successfully! Saved as `{plot_path}`."
    except Exception as e:
        return f"❌ Error executing plot code: {str(e)}"

# Function to handle user queries
def data_analysis_assistant(message, history):
    global df_global

    if agent is None or df_global is None:
        return "⚠ Please upload a valid CSV file first."

    # Predefined introduction and question list
    intro = """Hello! I'm a data analysis assistant. I can answer queries based on your CSV file.

                You can ask questions like:
                1. What are the column names and their data types?
                2. What is the total number of rows and columns?
                3. What are the unique values in all columns?
                4. What is the average value of a numeric column?
                5. Are there any missing values in the dataset?
                6. Show me the first 5 rows of the dataset.
                7. Show me the basic information about the dataset.

                Enter the question number (1-7) or type your own question:"""

    if message.lower() in ["hello", "hi", "start"]: # Initialize introduction
        return intro

    # Map user input numbers to predefined queries
    question_mapping = {
        "1": "column names and data types",
        "2": "Number of rows and columns",
        "3": "Unique values in all column",
        "4": "Describe all numerical columns",
        "5": "Missing values in dataset",
        "6": "Head of dataset",
        "7": "Basic information about dataset"
    }

    query = question_mapping.get(message, message)  # Default to user query

    # Keywords to detect plot requests
    plot_keywords = ["plot", "distribution", "graph", "visualization", "histogram", "scatter", "chart", "bar", "correlation"]

    if any(keyword in message.lower() for keyword in plot_keywords):
        try:
            # Ask the AI to generate only raw Python code
            response = agent.invoke(f"Generate ONLY valid Python code (no explanations, no markdown, no extra text) to plot {message} using Seaborn or Matplotlib. Assume `df` is the dataframe. Do NOT include `plt.show()`.")

            # Extract only the Python code
            plot_code = extract_code(response["output"])

            # Execute the extracted plot code
            return execute_plot_code(plot_code)
        except Exception as e:
            return f"❌ Error processing LLM response: {str(e)}"

    # Answer normal text queries
    response = agent.invoke(query)
    return response["output"]

# Define Gradio interface
with gr.Blocks(css="""
    body {
        background-color: #1e1e2e !important;
        color: white !important;
    }
    
    .gradio-container {
        background-color: #1e1e2e !important;
    }
    h1, h2, h3, h4, h5 {
        color: #ffcc00 !important;
    }
""") as demo:

    gr.Markdown("# 📊 CSV Data Analysis Chatbot")
    
    file_upload = gr.File(label="Upload your CSV file", type="filepath")
    upload_button = gr.Button("Process File")
    upload_status = gr.Textbox(label="Upload Status", interactive=False)
    
    chatbot = gr.ChatInterface(fn=data_analysis_assistant, type="messages")
    
    upload_button.click(fn=create_agent, inputs=file_upload, outputs=upload_status)

# Launch the app
demo.launch()
