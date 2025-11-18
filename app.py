import pandas as pd
import matplotlib.pyplot as plt
from langchain.tools import tool
from langchain_groq import ChatGroq
import streamlit as st

import os

st.title("AI Graph Generator (Groq + LangChain)")
uploaded = st.file_uploader("Upload your CSV", type=["csv"])

# Load your dataset
df1 = pd.read_csv('CPI by Year.csv')

if uploaded:
    df1 = pd.read_csv(uploaded)
    st.write("### Data Preview", df1.head())

    # 1️⃣ Define a Python execution tool
    @tool
    def python_repl(code: str) -> str:
        """
        Execute Python code using df and matplotlib.
        Must call plt.show() if generating graphs.
        """
        local_vars = {"df1": df1, "plt": plt}
        try:
            exec(code, {}, local_vars)
            return "Execution successful."
        except Exception as e:
            return f"Error: {e}"
    
    
    # 2️⃣ LLM (Groq Llama 3 70B)
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
    ).bind_tools([python_repl])

    prompt = st.text_area("Write a graph prompt")

    if st.button("Generate Graph"):
        system = """
        You are a data visualization AI. Use matplotlib. Always call python_repl
        with Python code and include plt.show().
        """


# # 3️⃣ Function to run prompt → graph
# def generate_graph(prompt: str):
#     system = """
#     You are a data visualization assistant.
#     You have access to a pandas DataFrame called `df1`.
#     Only write Python code and always call python_repl with it.
#     Use matplotlib for all plots.
#     Use plt.show() to display graphs.
#     """

    # Ask LLM to produce Python code
    msg = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    )

    # Handle tool call
    if msg.tool_calls:
        for call in msg.tool_calls:
            python_repl.run(call["args"]["code"])
            # if call["name"] == "python_repl":
            #     code = call["args"]["code"]
            #     print("\n--- Running Code ---\n")
            #     print(code)
            #     print("\n--- Output ---\n")
            #     return python_repl.run(code)

    return msg.content


# 4️⃣ Example
if __name__ == "__main__":
    generate_graph("Plot Year vs Food, F&V, and Sugar as a multi-line chart.")
