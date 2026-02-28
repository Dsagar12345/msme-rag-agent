import torch
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

def test_gpu():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No GPU detected — using CPU (this is fine)")

def test_chromadb():
    client = chromadb.Client()
    client.create_collection("test")
    print("ChromaDB working ✅")

def test_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Gemini API key not found ❌ — check your .env file")
        return
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content("Say exactly: setup working")
    print(f"Gemini API working ✅ — {response.text}")

if __name__ == "__main__":
    print("="*40)
    print("MSME RAG Agent — Setup Test")
    print("="*40)
    test_gpu()
    test_chromadb()
    test_gemini()
    print("="*40)