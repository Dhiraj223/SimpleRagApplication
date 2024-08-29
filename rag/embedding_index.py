from typing import List, Dict, Tuple, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class VectorStore:
    """
    A class for storing, indexing, and retrieving text data using vector embeddings,
    optimized for Retrieval-Augmented Generation (RAG).
    """

    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the VectorStore.

        Args:
            embedding_model_name (str): The name of the sentence transformer model to use for embeddings.
        """
        self.embeddings = []
        self.chunks = []
        self.model = SentenceTransformer(embedding_model_name)

    def add_texts(self, chunks: List[Dict[str, Union[str, int]]]) -> None:
        """
        Add text chunks to the vector store.

        Args:
            chunks (List[Dict[str, Union[str, int]]]): A list of text chunks with their metadata.
        """
        for chunk in chunks:
            embedding = self.model.encode(chunk['text'])
            self.embeddings.append(embedding)
            self.chunks.append(chunk)

    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Dict[str, Union[str, int]], float]]:
        """
        Perform a similarity search for the given query.

        Args:
            query (str): The search query.
            k (int): The number of top results to return.

        Returns:
            List[Tuple[Dict[str, Union[str, int]], float]]: A list of tuples containing the top-k similar chunks and their similarity scores.
        """
        query_embedding = self.model.encode(query)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        top_k_indices = np.argsort(similarities)[-k:][::-1]

        return [(self.chunks[i], similarities[i]) for i in top_k_indices]

    def get_relevant_documents(self, query: str, k: int = 5) -> List[Dict[str, Union[str, int]]]:
        """
        Retrieve the most relevant documents for a given query.

        Args:
            query (str): The search query.
            k (int): The number of documents to retrieve.

        Returns:
            List[Dict[str, Union[str, int]]]: A list of the most relevant text chunks with their metadata.
        """
        results = self.similarity_search(query, k)
        return [chunk for chunk, _ in results]