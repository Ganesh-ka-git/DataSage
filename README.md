# DataSage

Traditional data analysis demands specialized expertise, restricting access to insights for non-technical users. By developing an AI agent that processes natural language queries, we can make data analysis accessible to everyone. This empowers users to extract meaningful insights from complex datasets without requiring programming or statistical knowledge.

The primary goal of this project is to create an intelligent chatbot that can:
•	Understand user queries related to a given data file.
•	Perform data analysis using Pandas and LangChain.
•	 Provide insights based on user requests.
•	 Offer an interactive and user-friendly interface through Gradio.

Technologies & Tools Used
To build this chatbot, we utilized the following technologies:
1.	GPT Models: Experimented with GPT-4o, GPT-3.5, and other versions to determine the best performance.
2.	Pandas: Python Library for data manipulation and analysis.
3.	LangChain: High Level Python Library to enhance chatbot capabilities with structured query processing.
4.	Gradio: For developing an interactive and easy-to-use UI for users.

Implementation Details
The chatbot is designed to analyze data files based on user queries. It follows these steps:
•	 The user uploads a data file.
•	 The chatbot processes the file using Pandas and extracts relevant insights.
•	 Based on the user’s query, the chatbot applies appropriate data transformations and analysis.
•	 The response is generated and presented through the Gradio interface


