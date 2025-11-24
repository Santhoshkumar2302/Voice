import os, uuid, base64, asyncio, pandas as pd, time
from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# LangChain + embeddings + RAG
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# TTS
import edge_tts

# -------------------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
templates = Jinja2Templates(directory="templates")
os.makedirs("static", exist_ok=True)

# ------------------- Embeddings + Vector DB
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
DB_DIR = "./support_db"
if not os.path.exists(DB_DIR):
    df=pd.read_csv("customer_faq.csv",encoding="utf-8")
    loader=DataFrameLoader(df,page_content_column="Answer")
    docs=loader.load()
    vectorstore=Chroma.from_documents(docs,hf_embeddings,persist_directory=DB_DIR)
    vectorstore.persist()
else:
    vectorstore=Chroma(persist_directory=DB_DIR,embedding_function=hf_embeddings)

# ------------------- Gemini LLM for RAG
llm=ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")
prompt_template = PromptTemplate(
    input_variables=["context","question"], 
    template="""
You are SupportAssistant, a professional and polite customer support voice bot.

Answer the USER QUESTION **only using the CONTEXT below**.  
Do not make up information.  

If the answer is not in the CONTEXT, respond politely with one of the following (choose any one):
- "I'm sorry, but I couldn't find that information in the provided details."
- "I apologize, but the context doesn't mention anything about that."
- "Thank you for your question! However, I don’t have that information in the given context."

CONTEXT:
{context}

USER QUESTION:
{question}

Answer:
"""
)

qa_chain=RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k":4}),
    chain_type_kwargs={"prompt":prompt_template}
)

# ------------------- TTS helper
async def text_to_speech_base64(text:str, voice:str="en-US-AriaNeural")->str:
    file_name=f"{uuid.uuid4().hex}.mp3"
    path=os.path.join("static",file_name)
    await edge_tts.Communicate(text=text,voice=voice).save(path)
    audio_bytes=await asyncio.to_thread(lambda: open(path,"rb").read())
    try: os.remove(path)
    except: pass
    return base64.b64encode(audio_bytes).decode()

# ------------------- Routes
@app.get("/")
def home(request:Request):
    return templates.TemplateResponse("index.html",{"request":request})

@app.post("/chat")
async def chat(message:str = Form(...)):
    try:
        start_total = time.perf_counter()

        start_llm = time.perf_counter()
        rag_answer = await asyncio.to_thread(qa_chain.run, message)
        end_llm = time.perf_counter()
        llm_time = round((end_llm - start_llm)*1000,2)

        audio_b64 = await text_to_speech_base64(rag_answer)

        end_total = time.perf_counter()
        total_time = round((end_total - start_total) * 1000,2)

        return {"reply": rag_answer, "audio_base64": audio_b64,"llm_time":llm_time,"total_time":total_time}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/tts")
async def tts(text:str=Form(...)):
    try:
        audio_b64 = await text_to_speech_base64(text)
        return {"audio_base64": audio_b64}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ------------------- Static mount
app.mount("/static", StaticFiles(directory="static"), name="static")
