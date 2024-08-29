import torch
from typing import List, Dict
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class Generator:
    """
    A class for generating answers based on retrieved documents.
    """

    def __init__(self, model_name: str = 'microsoft/Phi-3.5-mini-instruct'):
        """
        Initialize the Generator.

        Args:
            model_name (str): The name of the language model to use for generation.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        self.model.to(device)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        self.generation_args = {
            "max_new_tokens": 200,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False
        }

    def generate_answer(self, query: str, retrieved_docs: List[Dict[str, str]]) -> str:
        """
        Generate an answer based on the query and retrieved documents.

        Args:
            query (str): The user's question.
            retrieved_docs (List[Dict[str, str]]): A list of retrieved relevant documents.

        Returns:
            str: The generated answer.
        """
        context = " ".join([doc['text'] for doc in retrieved_docs])
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant that provides concise answers based on the provided context. Respond with only the answer to the question, without adding any extra explanation or text."},
            {"role": "user", "content": prompt},
        ]

        try:
            output = self.pipe(messages, **self.generation_args)
            return output[0]['generated_text']
        except Exception as e:
            return f"An error occurred while generating the answer: {str(e)}"