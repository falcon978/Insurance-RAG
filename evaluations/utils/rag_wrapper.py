# evaluations/utils/rag_wrapper.py
import os
from engine import InsuranceRAGEngine
from config import settings

class EvalRAGWrapper:
    """
    Wraps the InsuranceRAGEngine to expose both the final LLM response 
    and the retrieved nodes required by DeepEval metrics.
    """
    def __init__(self):
        # Initialize the core engine using the central config
        self.engine = InsuranceRAGEngine(gemini_api_key=settings.gemini_api_key)

    def query(self, query: str, collection_name: str, retrieve_top_k: int, rerank_top_k: int, disable_bm25: bool = False):
        """
        Executes the RAG pipeline component-by-component to capture context.
        """
        # 1. Retrieval Phase
        # Pass disable_bm25 if your retriever supports isolating vector search
        kwargs = {"query": query, "collection_name": collection_name, "top_k": retrieve_top_k}
        if disable_bm25:
            kwargs["use_hybrid"] = False # Adjust this parameter based on your retriever's exact signature
            
        raw_nodes = self.engine.retriever.retrieve(**kwargs)
        
        # 2. Reranking Phase
        if rerank_top_k > 0:
            final_nodes = self.engine.reranker.rerank(query, raw_nodes, top_k=rerank_top_k)
        else:
            final_nodes = raw_nodes[:retrieve_top_k]
            
        # Extract plain text for DeepEval (adjust .text or .page_content based on your Node structure)
        retrieved_contexts = [getattr(node, 'text', str(node)) for node in final_nodes]
        
        # 3. Generation Phase
        actual_output = self.engine.generator.generate(
            query=query, 
            contexts=retrieved_contexts, 
            history=[]
        )
        
        return actual_output, retrieved_contexts