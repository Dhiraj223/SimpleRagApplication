from rag import DocumentLoader, DocumentSplitter, VectorStore, Generator

def main():
    try:
        # Load data
        loader = DocumentLoader("data/test.txt")
        data = loader.load_data()

        # Split data
        splitter = DocumentSplitter(data)
        chunks = splitter.split_data()

        # Create and populate the embedding index
        index = VectorStore()
        index.add_texts(chunks)

        # Initialize generator
        generator = Generator()

        # Example query
        query = "What is the main topic of this document?"

        # Retrieve relevant chunks
        relevant_chunks = index.get_relevant_documents(query)

        # Generate answer
        answer = generator.generate_answer(query, relevant_chunks)

        print(f"Query: {query}")
        print(f"Answer: {answer}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()