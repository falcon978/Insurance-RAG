"""
custom_judge.py
---------------
Provides a globally accessible DeepEval judge wrapper.
Automatically configures itself using eval_config.py.
"""
from langchain_groq import ChatGroq
from deepeval.models.base_model import DeepEvalBaseLLM
from evaluations.eval_config import eval_settings

class GroqJudge(DeepEvalBaseLLM):
    """DeepEval wrapper for Groq-hosted models."""
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        res = await self.model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return eval_settings.judge_model_name

def get_eval_judge() -> GroqJudge:
    """
    Factory function to instantiate and return the configured judge.
    Ensures that every test file uses the exact same evaluator model.
    """
    groq_model = ChatGroq(
        model=eval_settings.judge_model_name, 
        groq_api_key=eval_settings.groq_api_key
    )
    return GroqJudge(model=groq_model)