import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pymongo import MongoClient
import datetime
from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")
mango_uri=os.getenv("MANGO_URI")

client =MongoClient(mango_uri)
db= client["chat"]
collection =db["users"]

app=FastAPI()

class ChatRequest(BaseModel):
    user_id:str
    question:str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI Study Assistant. Answer academic and learning-related questions clearly and simply."),
        ("placeholder", "{history}"),
        ("user", "{question}")
    ]
)

llm= ChatGroq(api_key=groq_api_key, model="openai/gpt-oss-20b")
chain=prompt | llm


def get_history(user_id):
    chats = collection.find(
        {"user_id": user_id}
    ).sort("timestamp", -1).limit(10)

    history = []
    for chat in reversed(list(chats)):
        history.append((chat["role"], chat["message"]))

    return history

@app.get("/")
def home():
    return {
        "project": "Study Bot",
        "description": "AI-powered Study Assistant with MongoDB memory",
        "status": "Running Successfully"
    }

@app.post("/chat")
def chat(chat_request: ChatRequest):

    history = get_history(chat_request.user_id)

    response = chain.invoke({
        "history": history,
        "question": chat_request.question
    })

    collection.insert_one({
        "user_id": chat_request.user_id,
        "role": "user",
        "message": chat_request.question,
        "timestamp": datetime.datetime.utcnow()
    })

    collection.insert_one({
        "user_id": chat_request.user_id,
        "role": "assistant",
        "message": response.content,
        "timestamp": datetime.datetime.utcnow()
    })

    return {"response": response.content}