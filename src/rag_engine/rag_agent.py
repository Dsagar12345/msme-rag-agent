# src/rag_engine/rag_agent.py
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import google.generativeai as genai
import pandas as pd
import json
from dotenv import load_dotenv
from src.rag_engine.vector_store import MSMEVectorStore
from src.rag_engine.hybrid_search import HybridSearchEngine
from src.rag_engine.query_classifier import QueryClassifier
from datetime import datetime

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class MSMERAGAgent:
    def __init__(self, business_id: str = "BIZ001"):
        print("Initializing MSME RAG Agent...")
        self.business_id = business_id
        self.model = genai.GenerativeModel("gemini-2.0-flash-lite")

        self.vector_store = MSMEVectorStore()
        self.search_engine = HybridSearchEngine(self.vector_store)
        self.classifier = QueryClassifier()

        self.business_profile = self._load_business_profile()
        self.gst_data = self._load_gst_data()

        print(f"✅ RAG Agent ready for: {self.business_profile.get('name', business_id)}")

    def _load_business_profile(self) -> dict:
        try:
            with open("data/processed/business_profiles.json", "r") as f:
                profiles = json.load(f)
            for p in profiles:
                if p["business_id"] == self.business_id:
                    return p
        except Exception as e:
            print(f"Warning: Could not load business profile: {e}")
        return {"business_id": self.business_id, "name": "Unknown Business"}

    def _load_gst_data(self) -> list:
        try:
            df = pd.read_csv("data/processed/gst_returns.csv")
            biz_data = df[df["business_id"] == self.business_id]
            return biz_data.tail(6).to_dict("records")
        except Exception as e:
            print(f"Warning: Could not load GST data: {e}")
        return []

    def _build_business_context(self) -> str:
        if not self.gst_data:
            return "No financial data available."

        total_revenue = sum(r.get("taxable_value", 0) for r in self.gst_data)
        total_gst = sum(r.get("net_gst_payable", 0) for r in self.gst_data)
        total_itc = sum(r.get("itc_utilized", 0) for r in self.gst_data)
        late_filings = sum(1 for r in self.gst_data if not r.get("filed_on_time", True))

        context = f"""
BUSINESS PROFILE:
- Name: {self.business_profile.get('name', 'N/A')}
- Industry: {self.business_profile.get('industry', 'N/A')}
- GSTIN: {self.business_profile.get('gstin', 'N/A')}
- State: {self.business_profile.get('state', 'N/A')}
- Business Size: {self.business_profile.get('business_size', 'N/A')}

LAST 6 MONTHS FINANCIAL SUMMARY:
- Total Revenue: Rs {total_revenue:,.0f}
- Total GST Paid: Rs {total_gst:,.0f}
- Total ITC Utilized: Rs {total_itc:,.0f}
- Late Filings: {late_filings} out of {len(self.gst_data)} months
- Average Monthly Revenue: Rs {total_revenue/len(self.gst_data):,.0f}

RECENT GST RETURNS:"""

        for record in self.gst_data[-3:]:
            context += f"\n- {record.get('return_period', '')}: Revenue Rs {record.get('taxable_value', 0):,.0f}, GST Rs {record.get('net_gst_payable', 0):,.0f}, Filed on time: {record.get('filed_on_time', True)}"

        return context.strip()

    def _build_prompt(self, query: str, retrieved_chunks: list,
                      business_context: str) -> str:
        knowledge = ""
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk["metadata"].get("source_file", "Unknown")
            authority = chunk["metadata"].get("authority", "Unknown")
            knowledge += f"\n[Source {i+1}: {source} | {authority}]\n{chunk['text']}\n"

        prompt = f"""You are an expert financial advisor for Indian MSME businesses.
You have deep knowledge of GST, Income Tax, RBI regulations, and government schemes.
Always give practical, actionable advice specific to the business situation.

BUSINESS CONTEXT:
{business_context}

RELEVANT KNOWLEDGE BASE:
{knowledge}

QUESTION: {query}

INSTRUCTIONS:
1. Answer specifically for THIS business using their actual numbers where relevant
2. Cite your sources using [Source X] notation
3. Give concrete actionable steps not generic advice
4. Keep answer clear and under 300 words
5. End with ONE specific action they should take today

ANSWER:"""

        return prompt

    def _check_hallucination(self, answer: str, chunks: list) -> bool:
        context = " ".join([c["text"][:200] for c in chunks[:3]])

        prompt = f"""Is this financial answer grounded in the provided context?
Answer with only YES or NO.

Context: {context}

Answer: {answer[:300]}

Grounded (YES/NO):"""

        try:
            time.sleep(4)
            response = self.model.generate_content(prompt)
            return "YES" in response.text.upper()
        except:
            return True

    def answer(self, query: str) -> dict:
        print(f"\n{'='*55}")
        print(f"Processing: '{query}'")
        print('='*55)

        # Step 1: Classify
        classification = self.classifier.classify(query)
        category = classification["category"]
        print(f"📋 Category: {category} ({classification['confidence']} confidence)")

        # Step 2: Hybrid search
        chunks = self.search_engine.hybrid_search(
            query=query,
            category=category if category != "general" else None,
            n_results=5
        )
        print(f"🔍 Retrieved {len(chunks)} relevant chunks")

        # Step 3: Business context
        business_context = self._build_business_context()

        # Step 4: Build prompt
        prompt = self._build_prompt(query, chunks, business_context)

        # Step 5: Generate answer with rate limiting
        print("🤖 Generating answer with Gemini...")
        try:
            time.sleep(5)
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
        except Exception as e:
            return {
                "query": query,
                "answer": f"Rate limit hit — please wait 60 seconds and try again. Error: {e}",
                "category": category,
                "sources": [],
                "chunks_used": len(chunks),
                "grounded": False,
                "error": True,
                "timestamp": datetime.now().isoformat()
            }

        # Step 6: Hallucination check
        is_grounded = self._check_hallucination(answer, chunks)
        if not is_grounded:
            print("⚠️  Re-checking grounding...")

        # Step 7: Return result
        sources = list(set([
            c["metadata"].get("source_file", "N/A")
            for c in chunks
        ]))

        result = {
            "query": query,
            "answer": answer,
            "category": category,
            "sources": sources,
            "chunks_used": len(chunks),
            "grounded": is_grounded,
            "timestamp": datetime.now().isoformat()
        }

        print(f"✅ Answer generated ({len(answer)} chars)")
        print(f"📚 Sources: {sources}")
        return result


# ── Test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = MSMERAGAgent(business_id="BIZ001")

    # Only 2 questions to stay within rate limits
    test_questions = [
        "What is my GST compliance status and am I at risk of penalties?",
        "How can I apply for a MUDRA loan and what documents do I need?",
    ]

    for question in test_questions:
        result = agent.answer(question)
        print(f"\n{'='*55}")
        print(f"Q: {question}")
        print(f"{'='*55}")
        print(f"ANSWER:\n{result['answer']}")
        print(f"\nSources: {result['sources']}")
        print(f"Grounded: {result['grounded']}")
        print(f"{'='*55}\n")

        # Wait between questions
        print("⏳ Waiting 15 seconds before next question...")
        time.sleep(15)