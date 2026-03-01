# src/rag_engine/rag_agent.py
import os
import sys
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
genai.configure(api_key=os.getenv("GENAI_API_KEY"))


class MSMERAGAgent:
    """
    The core RAG agent that answers financial questions.
    
    Flow:
    1. Classify query → which collection to search
    2. Hybrid search → retrieve best chunks
    3. Load business context → their actual numbers
    4. Build prompt → combine query + chunks + business data
    5. Generate answer → Gemini API
    6. Verify answer → hallucination check
    7. Return cited answer
    """

    def __init__(self, business_id: str = "BIZ001"):
        print("Initializing MSME RAG Agent...")
        self.business_id = business_id
        self.model = genai.GenerativeModel("gemini-2.5-flash")

        # Initialize components
        self.vector_store = MSMEVectorStore()
        self.search_engine = HybridSearchEngine(self.vector_store)
        self.classifier = QueryClassifier()

        # Load business data
        self.business_profile = self._load_business_profile()
        self.gst_data = self._load_gst_data()

        print(f"✅ RAG Agent ready for: {self.business_profile.get('name', business_id)}")

    def _load_business_profile(self) -> dict:
        """Load business profile from JSON"""
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
        """Load last 6 months of GST data for business"""
        try:
            df = pd.read_csv("data/processed/gst_returns.csv")
            biz_data = df[df["business_id"] == self.business_id]
            return biz_data.tail(6).to_dict("records")
        except Exception as e:
            print(f"Warning: Could not load GST data: {e}")
        return []

    def _build_business_context(self) -> str:
        """
        Creates a financial summary of the business.
        This gets injected into every prompt so Gemini
        knows WHO it's talking to and their actual numbers.
        """
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

RECENT GST RETURNS:
"""
        for record in self.gst_data[-3:]:
            context += f"- {record.get('return_period', '')}: Revenue Rs {record.get('taxable_value', 0):,.0f}, GST Rs {record.get('net_gst_payable', 0):,.0f}, Filed on time: {record.get('filed_on_time', True)}\n"

        return context.strip()

    def _build_prompt(self, query: str, retrieved_chunks: list,
                      business_context: str) -> str:
        """
        Builds the final prompt sent to Gemini.
        Structure:
        1. System role
        2. Business context (their actual numbers)
        3. Retrieved knowledge (regulations, guides)
        4. The question
        5. Instructions for answering
        """
        # Format retrieved chunks
        knowledge = ""
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk["metadata"].get("source_file", "Unknown")
            authority = chunk["metadata"].get("authority", "Unknown")
            knowledge += f"\n[Source {i+1}: {source} | {authority}]\n{chunk['text']}\n"

        prompt = f"""You are an expert financial advisor for Indian MSME businesses.
You have deep knowledge of GST, Income Tax, RBI regulations, and government schemes.
Always give practical, actionable advice specific to the business's situation.

BUSINESS CONTEXT:
{business_context}

RELEVANT KNOWLEDGE BASE:
{knowledge}

QUESTION: {query}

INSTRUCTIONS:
1. Answer specifically for THIS business using their actual numbers where relevant
2. Cite your sources using [Source X] notation
3. Give concrete actionable steps, not generic advice
4. If the question involves their compliance status, mention it
5. Keep answer clear and under 300 words
6. End with ONE specific action they should take today

ANSWER:"""

        return prompt

    def _check_hallucination(self, answer: str, chunks: list) -> bool:
        """
        Quick hallucination check.
        Asks Gemini if the answer is grounded in retrieved chunks.
        Returns True if answer is grounded, False if hallucinated.
        """
        context = " ".join([c["text"][:200] for c in chunks[:3]])

        prompt = f"""Is this financial answer grounded in the provided context?
Answer with only YES or NO.

Context: {context}

Answer: {answer[:300]}

Grounded (YES/NO):"""

        try:
            response = self.model.generate_content(prompt)
            return "YES" in response.text.upper()
        except:
            return True  # Assume grounded if check fails

    def answer(self, query: str) -> dict:
        """
        Main method — takes a question, returns a full answer.
        This is what gets called from the UI.
        """
        print(f"\n{'='*55}")
        print(f"Processing: '{query}'")
        print('='*55)

        # Step 1: Classify query
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

        # Step 3: Load business context
        business_context = self._build_business_context()

        # Step 4: Build prompt
        prompt = self._build_prompt(query, chunks, business_context)

        # Step 5: Generate answer
        print("🤖 Generating answer with Gemini...")
        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
        except Exception as e:
            answer = f"Error generating answer: {e}"
            return {"answer": answer, "error": True}

        # Step 6: Hallucination check
        is_grounded = self._check_hallucination(answer, chunks)
        if not is_grounded:
            print("⚠️  Hallucination detected — regenerating...")
            response = self.model.generate_content(prompt)
            answer = response.text.strip()

        # Step 7: Build response object
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

    test_questions = [
        "What is my GST compliance status and am I at risk of penalties?",
        "How can I apply for a MUDRA loan and what documents do I need?",
        "What is input tax credit and how much ITC can I claim?",
        "Can I restructure my business loan given current stress?"
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