"""
rag_wrapper.py
--------------
Wraps the core InsuranceRAGEngine to execute the pipeline step-by-step.
Allows DeepEval to capture intermediate contexts and natively overrides 
the retrieval strategy (hybrid vs semantic).
"""
import os
from engine import InsuranceRAGEngine
from config import settings

class EvalRAGWrapper:
    def __init__(self):
        self.engine = InsuranceRAGEngine(gemini_api_key=settings.gemini_api_key)

    def _get_collection_name(self, source: str) -> str:
        """Safely maps the dataset source to your actual Chroma collections."""

        return f"insurance_{source}"

    def query(self, query: str, source: str, retrieve_top_k: int, rerank_top_k: int, strategy: str = "hybrid"):
        
        # --- Handle Comparison Queries ---
        if source == "both":
            col_a = self._get_collection_name("care_supreme")
            col_b = self._get_collection_name("optima_secure")

            # Fetch from both
            eng_a = self.engine._get_search_engine(col_a, retrieve_top_k)
            eng_b = self.engine._get_search_engine(col_b, retrieve_top_k)

            # Apply strategy overrides to BOTH engines
            if eng_a.strategy != strategy:
                eng_a.strategy = strategy
                eng_a.retriever = eng_a._initialize_strategy()
            if eng_b.strategy != strategy:
                eng_b.strategy = strategy
                eng_b.retriever = eng_b._initialize_strategy()
            
            # Fetch from both
            nodes_a = eng_a.search(query)
            nodes_b = eng_b.search(query)
            
            # Rerank both
            if rerank_top_k > 0:
                nodes_a = self.engine.reranker.rerank(query, nodes_a, top_k=rerank_top_k)
                nodes_b = self.engine.reranker.rerank(query, nodes_b, top_k=rerank_top_k)
            else:
                nodes_a = nodes_a[:retrieve_top_k]
                nodes_b = nodes_b[:retrieve_top_k]
                
            retrieved_contexts = [node.page_content for node in nodes_a + nodes_b]
            actual_output = self.engine.compare_policies(
                query=query, collection_a=col_a, collection_b=col_b, 
                retrieve_top_k=retrieve_top_k, rerank_top_k=rerank_top_k
            )
            return actual_output, retrieved_contexts

        # --- Handle Single Policy Queries ---
        collection_name = self._get_collection_name(source)
        search_engine = self.engine._get_search_engine(collection_name, retrieve_top_k)
        
        # Natively override the strategy if evaluating pure semantic search
        if search_engine.strategy != strategy:
            search_engine.strategy = strategy
            search_engine.retriever = search_engine._initialize_strategy()
            
        raw_nodes = search_engine.search(query)
        
        # Reranking Phase
        if rerank_top_k > 0:
            final_nodes = self.engine.reranker.rerank(query, raw_nodes, top_k=rerank_top_k)
        else:
            final_nodes = raw_nodes[:retrieve_top_k]
            
        # Extract plain text STRICTLY for DeepEval metrics to read
        retrieved_contexts = [node.page_content for node in final_nodes]
        
        # Generation Phase
        # Pass the actual Document objects to the generator (as expected by engine.py)
        actual_output = self.engine.generator.generate_single_answer(
            query=query,
            docs=final_nodes, 
            policy_name=self.engine._format_policy_name(collection_name)
        )
        
        return actual_output, retrieved_contexts