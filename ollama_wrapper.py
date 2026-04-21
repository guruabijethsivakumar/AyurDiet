from deepeval.models.llms.openai_model import GPTModel
from langchain_community.llms import Ollama

class OllamaModel(GPTModel):
    """
    DeepEval-compatible wrapper for Ollama local LLMs.
    Uses GPTModel as the base class because DeepEval v1.x
    still depends on this class for interface consistency.
    """

    def __init__(self, model_name="mistral:7b"):
        # Initialize GPTModel with a dummy model name
        super().__init__(model="ollama-" + model_name)

        self.model_name = model_name
        self.client = Ollama(model=model_name)

    def generate(self, prompt: str) -> str:
        """DeepEval calls this to get the model output."""
        response = self.client.invoke(prompt)
        return response

    def get_model_name(self) -> str:
        return f"Ollama-{self.model_name}"
