import streamlit as st
from io import StringIO, BytesIO
from PIL import Image
from rag import DocumentLoader, DocumentSplitter, VectorStore, Generator
import os

# Streamlit App
def main():
    st.title("Document Query System with RAG")

    st.sidebar.header("Upload Document")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['pdf', 'docx', 'txt', 'csv', 'xlsx', 'html', 'jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        # Read the file
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Load the document
        loader = DocumentLoader(file_path)
        data = loader.load_data()
        
        # Split the data into chunks
        splitter = DocumentSplitter(data)
        chunks = splitter.split_data()
        
        # Create and populate the embedding index
        index = VectorStore()
        index.add_texts(chunks)
        
        # Initialize the generator
        generator = Generator()
        
        st.sidebar.header("Ask a Question")
        query = st.sidebar.text_input("Enter your query")

        if query:
            # Retrieve relevant chunks
            relevant_chunks = index.get_relevant_documents(query)

            # Generate an answer
            answer = generator.generate_answer(query, relevant_chunks)

            st.subheader("Answer")
            st.write(answer)

        # Clean up the uploaded file
        os.remove(file_path)

if __name__ == "__main__":
    main()
