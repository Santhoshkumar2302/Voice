import os
import uuid
import asyncio
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from gtts import gTTS

# FastAPI setup
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates and static
templates = Jinja2Templates(directory="templates")
os.makedirs("static", exist_ok=True)

# --- Hugging Face Embeddings ---
# You can choose any sentence-transformer model
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load or create vectorstore
if not os.path.exists("./support_db"):
    df = pd.read_csv("customer_faq.csv", encoding="utf-8")
    loader = DataFrameLoader(df, page_content_column="Answer")
    documents = loader.load()
    vectorstore = Chroma.from_documents(documents, hf_embeddings, persist_directory="./support_db")
    vectorstore.persist()
else:
    vectorstore = Chroma(persist_directory="./support_db", embedding_function=hf_embeddings)

# Gemini Pro LLM
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")

# Prompt template
support_prompt = """
You are SupportAssistant, a professional and polite customer support voice bot for AcmeCorp.

Your goal:
- Accurately and politely answer user FAQs and ticket updates.
- Only use the CONTEXT below.

CONTEXT:
{context}

USER QUERY:
{question}

INSTRUCTIONS:
1. Give a clear, factual, and friendly voice-ready answer.
2. Keep under 3 sentences.
3. Do not guess or make up information.
"""

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=support_prompt
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    chain_type_kwargs={"prompt": prompt_template}
)

# Routes
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(message: str = Form(...)):
    try:
        response = await asyncio.to_thread(qa_chain.run, message)

        # Generate TTG
        audio_filename = f"response_{uuid.uuid4().hex}.mp3"
        audio_path = os.path.join("static", audio_filename)
        tts = gTTS(text=response, lang="en", slow=False)
        tts.save(audio_path)

        # Keep latest 20 audio files
        audio_files = sorted(
            [f for f in os.listdir("static") if f.endswith(".mp3")],
            key=lambda x: os.path.getmtime(os.path.join("static", x))
        )
        for old_file in audio_files[:-20]:
            os.remove(os.path.join("static", old_file))

        return JSONResponse({"reply": response, "audio": f"/static/{audio_filename}"})

    except Exception as e:
        return JSONResponse({"error": str(e)})

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
