# src/rag_engine/query_classifier.py
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class QueryClassifier:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-2.0-flash-lite")

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
        query_lower = query.lower()
        scores = {category: 0 for category in self.keyword_map}

        for category, keywords in self.keyword_map.items():
            for keyword in keywords:
                if keyword in query_lower:
                    scores[category] += 1

        best_category = max(scores, key=scores.get)

        if scores[best_category] == 0:
            return "general"

        return best_category

    def classify_with_llm(self, query: str) -> dict:
        prompt = f"""You are a financial query classifier for Indian MSME businesses.

Classify this query into exactly ONE category:
- tax: GST, income tax, TDS, advance tax, tax filing, ITC
- regulatory: RBI guidelines, loans, NPA, restructuring, banking rules
- scheme: Government schemes like MUDRA, Udyam registration, subsidies
- research: Industry benchmarks, sector trends, market data
- financial: Business own financial data, their specific numbers
- general: Spans multiple categories or unclear

Query: "{query}"

Respond in this exact format:
CATEGORY: <category>
CONFIDENCE: <high/medium/low>
SEARCH_TERMS: <3-5 key terms to search for>
REASONING: <one sentence why>"""

        try:
            time.sleep(4)
            response = self.model.generate_content(prompt)
            text = response.text.strip()

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
            return {
                "category": self.classify_fast(query),
                "confidence": "low",
                "search_terms": query.split()[:5],
                "reasoning": f"Fallback to keyword: {e}"
            }

    def classify(self, query: str) -> dict:
        fast_category = self.classify_fast(query)

        if fast_category != "general":
            return {
                "category": fast_category,
                "confidence": "high",
                "search_terms": query.split()[:5],
                "reasoning": "Keyword-based classification",
                "method": "keyword"
            }

        llm_result = self.classify_with_llm(query)
        llm_result["method"] = "llm"
        return llm_result


if __name__ == "__main__":
    classifier = QueryClassifier()

    test_queries = [
        "How do I file my GSTR-3B return?",
        "Can I get a MUDRA loan if I have pending GST dues?",
        "What is the RBI rule for MSME loan restructuring?",
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