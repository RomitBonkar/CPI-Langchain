import base64
import io
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain.tools import tool
from langchain_groq import ChatGroq

app = FastAPI()

# Allow frontend apps to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "API is running"}

# -------- Tool to execute Python + return plot ----------
def fig_to_base64():
    """Convert current matplotlib figure to base64."""
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

@tool
def python_repl(code: str) -> str:
    """
    Executes Python code with `df` and `plt`.
    Must call plt.show() in the code.
    """
    global df
    local_vars = {"df": df, "plt": plt}
    try:
        exec(code, {}, local_vars)
        img_str = fig_to_base64()
        plt.clf()
        return img_str
    except Exception as e:
        return f"Error: {e}"

# 2️⃣ LLM (Groq Llama 3 70B)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
).bind_tools([python_repl])

# --------- API Endpoint -----------
@app.post("/generate-graph")
async def generate_graph(prompt: str = Form(...)):
    global df

    # Load your dataset
    #df = pd.read_csv('CPI by Year.csv')
    

    system = """
    You are a data visualization AI using matplotlib.
    ALWAYS return Python code and call the python_repl tool with it.
    Make sure the code calls plt.show().
    """

    msg = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    )

    # Handle tool call
    if msg.tool_calls:
        tc = msg.tool_calls[0]
        if tc["name"] == "python_repl":
            code = tc["args"]["code"]
            img_base64 = python_repl.run(code)
            return {"image": img_base64, "code": code}

    return {"error": msg.content}
