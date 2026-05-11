"""
rag_wrapper.py
--------------
Wraps the core InsuranceRAGEngine to execute the pipeline step-by-step.
Allows DeepEval to capture intermediate contexts and natively overrides
the retrieval strategy (hybrid vs semantic) via Reciprocal Rank Fusion weights.
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

    def query(
        self,
        query: str,
        source: str,
        retrieve_top_k: int,
        rerank_top_k: int,
        strategy: str = "hybrid",
    ):
        # --- STRATEGY OVERRIDE ---
        # If the test requests pure 'semantic', we mathematically mute the Lexical (BM25)
        # track by setting its RRF fusion weight to 0.0.
        s_weight = 1.0 if strategy == "semantic" else settings.default_semantic_weight
        l_weight = 0.0 if strategy == "semantic" else settings.default_lexical_weight

        # 1. Translate Query (Common for both Single and Comparison routes)
        structured_query = self.engine.rewriter_chain.invoke({"query": query})

        bm25_string = f"{query} {' '.join(structured_query.medical_terms)}".strip()
        vector_string = (
            f"{structured_query.canonical_query} "
            f"{' '.join(structured_query.expanded_terms)} "
            f"{' '.join(structured_query.exclusion_terms)} "
            f"{' '.join(structured_query.policy_sections)}"
        ).strip()

        combined_rerank_string = f"{bm25_string} {vector_string}"

        # ==========================================================
        # COMPARISON POLICIES
        # ==========================================================
        if source == "both":
            col_a = self._get_collection_name("care_supreme")
            col_b = self._get_collection_name("optima_secure")

            # Retrieve Both (Using weighted dual-track)
            fused_a = self.engine._dual_track_retrieve(
                query,
                bm25_string,
                vector_string,
                col_a,
                retrieve_top_k,
                s_weight,
                l_weight,
            )
            fused_b = self.engine._dual_track_retrieve(
                query,
                bm25_string,
                vector_string,
                col_b,
                retrieve_top_k,
                s_weight,
                l_weight,
            )

            # Rerank Both
            if rerank_top_k > 0:
                best_a = self.engine.reranker.rerank(
                    combined_rerank_string, fused_a, top_k=rerank_top_k
                )
                best_b = self.engine.reranker.rerank(
                    combined_rerank_string, fused_b, top_k=rerank_top_k
                )
            else:
                best_a = fused_a[:retrieve_top_k]
                best_b = fused_b[:retrieve_top_k]

            # Capture contexts for DeepEval
            retrieved_contexts = [doc.page_content for doc in best_a + best_b]

            # Pass DIRECTLY to generator so it uses the exact documents we captured
            actual_output = self.engine.generator.generate_comparison(
                query=query,
                docs_a=best_a,
                name_a=self.engine._format_policy_name(col_a),
                docs_b=best_b,
                name_b=self.engine._format_policy_name(col_b),
                history=[],
            )
            return actual_output, retrieved_contexts

        # ==========================================================
        # SINGLE POLICY
        # ==========================================================
        collection_name = self._get_collection_name(source)

        # Retrieve Single (Using weighted dual-track)
        fused_docs = self.engine._dual_track_retrieve(
            query,
            bm25_string,
            vector_string,
            collection_name,
            retrieve_top_k,
            s_weight,
            l_weight,
        )

        # Rerank
        if rerank_top_k > 0:
            best_docs = self.engine.reranker.rerank(
                combined_rerank_string, fused_docs, top_k=rerank_top_k
            )
        else:
            best_docs = fused_docs[:retrieve_top_k]

        # Capture contexts for DeepEval
        retrieved_contexts = [doc.page_content for doc in best_docs]

        # Pass DIRECTLY to generator so it uses the exact documents we captured
        actual_output = self.engine.generator.generate_single_answer(
            query=query,
            docs=best_docs,
            policy_name=self.engine._format_policy_name(collection_name),
            history=[],
        )

        return actual_output, retrieved_contexts
