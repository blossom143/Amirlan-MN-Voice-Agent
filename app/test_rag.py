"""
Verification script for Khan Bank Advanced RAG Pipeline
=======================================================
Tests Knowledge Loading, Chunking, Glossary Expansion, BM25,
RRF Retrieval, and RAG Triad benchmark.
"""

import os
import sys

# Ensure app directory is in Python path
curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from glossary import expand_query_with_banking_terms, extract_matched_glossary_metadata
from vector_store import LocalKnowledgeRetriever
from rag_evaluation import RAGTriadEvaluator


def run_tests():
    print("=" * 70)
    print("TESTING KHAN BANK ADVANCED RAG PIPELINE")
    print("=" * 70)

    # 1. Test Glossary Expansion
    test_queries = [
        "байрны зээлийн урьдчилгаа хэд вэ",
        "хагас сайнд ажиллах салбар",
        "карт залгичихлаа",
        "свифт код юу вэ"
    ]
    print("\n--- 1. Testing Colloquial Glossary Expansion ---")
    for q in test_queries:
        exp = expand_query_with_banking_terms(q)
        matches = extract_matched_glossary_metadata(q)
        cat = matches[0]["category"] if matches else "none"
        print(f"Original: '{q}'\nCategory: {cat}\nExpanded: '{exp}'\n")

    # 2. Test Knowledge Base Loader & Chunking
    print("--- 2. Testing Knowledge Base Loader & Chunking ---")
    retriever = LocalKnowledgeRetriever()
    print(f"Total Chunks Indexed: {len(retriever.chunks)}")
    for i, c in enumerate(retriever.chunks[:5]):
        print(f" Chunk {i+1} [{c['source_file']}]: Doc='{c['doc_title']}', Sec='{c['section_title']}' ({len(c['content'])} chars)")

    # 3. Test Retrieval Quality on Key Queries
    print("\n--- 3. Testing Hybrid Retrieval Quality ---")
    eval_queries = [
        ("Орон сууцны 6% зээл", "байрны зээлийн урьдчилгаа хэдэн хувь байдаг вэ?"),
        ("Цалингийн зээл", "Аппаар цалингийн зээл авахад ямар шимтгэлтэй вэ?"),
        ("Бямба гарагийн салбар", "Бямба гарагт ямар салбарууд ажилладаг вэ?"),
        ("SWIFT код", "Гадаадаас мөнгө хүлээн авахад свифт код юу вэ?"),
        ("Карт гацсан", "АТМ карт залгисан үед яаж буцааж авах вэ?"),
        ("Гүйлгээний лимит", "Khan Bank App-ын өдрийн гүйлгээний хязгаар хэд вэ?")
    ]

    for label, q in eval_queries:
        res = retriever.retrieve(q, top_k=2)
        print(f"\nQuery [{label}]: '{q}'")
        print(f" Method:  {res['retrieval_method']}")
        print(f" Matched: {res['matched_titles']}")
        print(f" URL:     {res['source_url']}")
        print(f" Preview: {res['verified_context'][:180].replace(chr(10), ' ')}...")

    # 4. Run RAG Triad Evaluator Benchmark
    print("\n--- 4. Running RAG Triad Evaluation Benchmark ---")
    evaluator = RAGTriadEvaluator(retriever=retriever)
    benchmark = evaluator.run_benchmark()
    summary = benchmark["benchmark_summary"]
    print(f"Total Cases:                 {summary['total_cases']}")
    print(f"Avg Context Precision:       {summary['avg_context_precision'] * 100:.1f}%")
    print(f"Avg Faithfulness:            {summary['avg_faithfulness'] * 100:.1f}%")
    print(f"Avg Answer Relevance:        {summary['avg_answer_relevance'] * 100:.1f}%")
    print(f"Composite RAG Triad Score:   {summary['composite_rag_triad_score'] * 100:.1f}%")
    print(f"Benchmark Duration:          {summary['benchmark_duration_secs']}s")

    print("\nCase Details:")
    for res in benchmark["detailed_results"]:
        print(f" - [{res['case_id']}]: Precision={res['context_precision']}, Faith={res['faithfulness']}, Rel={res['answer_relevance']} ({res['retrieval_latency_ms']}ms)")


    print("\nAll RAG Pipeline Tests Completed Successfully!")


if __name__ == "__main__":
    run_tests()
