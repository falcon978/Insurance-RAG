"""
rag_wrapper.py
--------------
Wraps the core InsuranceRAGEngine to execute the pipeline step-by-step.
Allows DeepEval to capture intermediate contexts and natively overrides
the retrieval strategy (hybrid vs semantic) via Reciprocal Rank Fusion weights.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from rag.factories import InferenceModelFactory
from rag.rerankers import ContextReranker
from rag.generator import ResponseGenerator
from rag.query_rewriter import get_structured_rewriter_chain
from engine import InsuranceRAGEngine
from config import settings


class EvalRAGWrapper:
    def __init__(
        self,
        vector_client=None,
        lexical_client=None,
    ):

        embeddings = HuggingFaceEmbeddings(
            model_name=settings.embed_model_name,
            model_kwargs={"device": settings.hf_device},
            encode_kwargs={"normalize_embeddings": True},
        )
        reranker = ContextReranker(
            model_name=settings.rerank_model_name, device=settings.hf_device
        )
        generator_llm = InferenceModelFactory.get_chat_model(
            model_name=settings.llm_model_name, temperature=0.0
        )
        generator = ResponseGenerator(llm=generator_llm)
        planner_llm = InferenceModelFactory.get_chat_model(
            model_name=settings.planner_model_name,
            temperature=settings.planner_temperature,
        )
        rewriter_chain = get_structured_rewriter_chain(planner_llm)

        self.engine = InsuranceRAGEngine(
            embeddings=embeddings,
            reranker=reranker,
            generator=generator,
            rewriter_chain=rewriter_chain,
            vector_client=vector_client,
            lexical_client=lexical_client,
        )

    def _get_collection_name(self, source: str) -> str:
        """Safely maps the dataset source to your actual Chroma collections."""
        return f"insurance_{source}"

    async def a_query(
        self,
        query: str,
        source: str,
        retrieve_top_k: int,
        rerank_top_k: int,
        strategy: str = "hybrid",
    ):
        # --- STRATEGY OVERRIDE ---
        # If the test requests pure 'semantic', the Lexical (BM25)
        # track is muted by setting its RRF fusion weight to 0.0.
        s_weight = 1.0 if strategy == "semantic" else settings.default_semantic_weight
        l_weight = 0.0 if strategy == "semantic" else settings.default_lexical_weight

        # 1. Translate Query
        structured_query = await self.engine.rewriter_chain.ainvoke({"query": query})

        bm25_string = f"{query} {' '.join(structured_query.medical_terms)}".strip()
        vector_string = (
            f"{structured_query.canonical_query} "
            f"{' '.join(structured_query.expanded_terms)} "
            f"{' '.join(structured_query.exclusion_terms)} "
            f"{' '.join(structured_query.policy_sections)}"
        ).strip()

        # ==========================================================
        # RETRIEVAL (Handles single vs both)
        # ==========================================================
        if source == "both":
            collection_a = "insurance_optima_secure"
            collection_b = "insurance_care_supreme"
            
            search_engine_a = self.engine._get_search_engine(
                collection_a, retrieve_top_k, s_weight, l_weight
            )
            search_engine_b = self.engine._get_search_engine(
                collection_b, retrieve_top_k, s_weight, l_weight
            )
            
            import asyncio
            fused_a, fused_b = await asyncio.gather(
                search_engine_a.a_search(query, bm25_string, vector_string),
                search_engine_b.a_search(query, bm25_string, vector_string),
            )
            
            combined_rerank_string = f"{bm25_string} {vector_string}"
            if rerank_top_k > 0:
                best_a, best_b = await asyncio.gather(
                    self.engine.reranker.a_rerank(combined_rerank_string, fused_a, rerank_top_k),
                    self.engine.reranker.a_rerank(combined_rerank_string, fused_b, rerank_top_k),
                )
            else:
                best_a, best_b = fused_a[:retrieve_top_k], fused_b[:retrieve_top_k]
                
            best_docs = best_a + best_b
            retrieved_contexts = [doc.page_content for doc in best_docs]
            
            actual_output = await self.engine.generator.a_generate_comparison(
                query=query,
                docs_a=best_a,
                name_a=self.engine._format_policy_name(collection_a),
                docs_b=best_b,
                name_b=self.engine._format_policy_name(collection_b),
                history=[],
            )
        else:
            collection_name = self._get_collection_name(source)
            search_engine = self.engine._get_search_engine(
                collection_name, retrieve_top_k, s_weight, l_weight
            )

            # Retrieve (Using weighted dual-track)
            fused_docs = await search_engine.a_search(
                original_query=query,
                bm25_string=bm25_string,
                vector_string=vector_string,
            )

            # Rerank
            combined_rerank_string = f"{bm25_string} {vector_string}"
            if rerank_top_k > 0:
                best_docs = await self.engine.reranker.a_rerank(
                    query=combined_rerank_string, documents=fused_docs, top_k=rerank_top_k
                )
            else:
                best_docs = fused_docs[:retrieve_top_k]

            # Capture contexts for DeepEval
            retrieved_contexts = [doc.page_content for doc in best_docs]

            # Pass DIRECTLY to generator
            actual_output = await self.engine.generator.a_generate_single_answer(
                query=query,
                docs=best_docs,
                policy_name=self.engine._format_policy_name(collection_name),
                history=[],
            )

        return actual_output, retrieved_contexts
