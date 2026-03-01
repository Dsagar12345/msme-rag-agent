# src/rag_engine/query_classifier.py
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class QueryClassifier:
    """
    Classifies incoming queries into categories BEFORE retrieval.
    This is critical — wrong category = wrong collection = wrong answer.
    
    Categories map directly to ChromaDB collections:
    - tax        → GST, Income Tax, TDS, advance tax
    - regulatory → RBI guidelines, loan rules, NPA
    - scheme     → MUDRA, Udyam, government schemes
    - research   → benchmarks, industry trends, SIDBI data
    - financial  → business's own GST/bank data
    - general    → broad questions spanning multiple categories
    """

    def __init__(self):
        self.model = genai.GenerativeModel("gemini-2.0-flash")

        # Keyword-based fast classification (no API call needed)
        self.keyword_map = {
            "tax": [
                "gst", "income tax", "tds", "advance tax", "itr", "section 44",
                "tax return", "cgst", "sgst", "igst", "input tax credit", "itc",
                "filing", "gstr", "tax rate", "deduction", "exemption", "cess"
            ],
            "regulatory": [
                "loan", "rbi", "npa", "restructure", "bank", "credit", "repay",
                "emi", "interest rate", "working capital", "overdraft", "default",
                "moratorium", "stress", "resolution", "priority sector"
            ],
            "scheme": [
                "mudra", "udyam", "scheme", "subsidy", "register", "registration",
                "certificate", "government scheme", "startup", "shishu", "kishore",
                "tarun", "pmmy", "msme scheme", "benefit", "eligibility"
            ],
            "research": [
                "benchmark", "industry average", "trend", "sector", "compare",
                "performance", "market", "growth rate", "industry", "sidbi"
            ],
            "financial": [
                "my gst", "my revenue", "my business", "my filing", "my bank",
                "my balance", "my expense", "my profit", "how much did i",
                "what did i pay", "my turnover", "last month", "this quarter"
            ]
        }

    def classify_fast(self, query: str) -> str:
        """
        Fast keyword-based classification.
        No API call — instant response.
        Used as first pass before LLM classification.
        """
        query_lower = query.lower()
        scores = {category: 0 for category in self.keyword_map}

        for category, keywords in self.keyword_map.items():
            for keyword in keywords:
                if keyword in query_lower:
                    scores[category] += 1

        best_category = max(scores, key=scores.get)

        # If no keywords matched, return general
        if scores[best_category] == 0:
            return "general"

        return best_category

    def classify_with_llm(self, query: str) -> dict:
        """
        LLM-based classification for ambiguous queries.
        Returns category + reasoning + suggested search terms.
        """
        prompt = f"""You are a financial query classifier for Indian MSME businesses.

Classify this query into exactly ONE category:
- tax: GST, income tax, TDS, advance tax, tax filing, ITC
- regulatory: RBI guidelines, loans, NPA, restructuring, banking rules  
- scheme: Government schemes like MUDRA, Udyam registration, subsidies
- research: Industry benchmarks, sector trends, market data
- financial: Business's own financial data, their specific numbers
- general: Spans multiple categories or unclear

Query: "{query}"

Respond in this exact format:
CATEGORY: <category>
CONFIDENCE: <high/medium/low>
SEARCH_TERMS: <3-5 key terms to search for>
REASONING: <one sentence why>"""

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()

            # Parse response
            lines = text.split('\n')
            result = {
                "category": "general",
                "confidence": "low",
                "search_terms": [],
                "reasoning": ""
            }

            for line in lines:
                if line.startswith("CATEGORY:"):
                    result["category"] = line.split(":", 1)[1].strip().lower()
                elif line.startswith("CONFIDENCE:"):
                    result["confidence"] = line.split(":", 1)[1].strip().lower()
                elif line.startswith("SEARCH_TERMS:"):
                    terms = line.split(":", 1)[1].strip()
                    result["search_terms"] = [t.strip() for t in terms.split(",")]
                elif line.startswith("REASONING:"):
                    result["reasoning"] = line.split(":", 1)[1].strip()

            return result

        except Exception as e:
            # Fallback to keyword classification
            return {
                "category": self.classify_fast(query),
                "confidence": "low",
                "search_terms": query.split()[:5],
                "reasoning": f"Fallback to keyword classification: {e}"
            }

    def classify(self, query: str) -> dict:
        """
        Main classification method.
        Uses fast keyword check first.
        If ambiguous, uses LLM for better accuracy.
        """
        # Fast classification first
        fast_category = self.classify_fast(query)

        # If fast classification is confident, use it
        if fast_category != "general":
            return {
                "category": fast_category,
                "confidence": "high",
                "search_terms": query.split()[:5],
                "reasoning": "Keyword-based classification",
                "method": "keyword"
            }

        # Ambiguous — use LLM
        llm_result = self.classify_with_llm(query)
        llm_result["method"] = "llm"
        return llm_result


# ── Test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    classifier = QueryClassifier()

    test_queries = [
        "How do I file my GSTR-3B return?",
        "Can I get a MUDRA loan if I have pending GST dues?",
        "What is the RBI rule for MSME loan restructuring?",
        "How does my revenue compare to industry benchmark?",
        "What was my GST payment last quarter?",
        "How can I grow my textile business?",
        "What is Section 44AD of Income Tax?",
        "How to register under Udyam portal?"
    ]

    print("="*55)
    print("Query Classifier Test")
    print("="*55)

    for query in test_queries:
        result = classifier.classify(query)
        print(f"\nQuery: '{query}'")
        print(f"  Category   : {result['category']}")
        print(f"  Confidence : {result['confidence']}")
        print(f"  Method     : {result['method']}")
        print(f"  Reasoning  : {result['reasoning']}")