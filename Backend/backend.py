import argparse
import os
import sys

from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Ask a question about a PDF file.")
    parser.add_argument("pdf", help="path to the PDF file")
    parser.add_argument("question", help="question to ask about the PDF")
    args = parser.parse_args()

    missing = [name for name in ("OPENAI_API_KEY", "PINECONE_API_KEY") if not os.getenv(name)]
    if missing:
        sys.exit(f"Missing environment variables: {', '.join(missing)}")

    # Pinecone index names must be lowercase
    index_name = os.getenv("PINECONE_INDEX_NAME", "chatpdf")

    # Load data
    data = PyMuPDFLoader(args.pdf).load()

    # Chunking data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(data)

    # Embed the chunks and store them in Pinecone
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore.from_texts(
        [t.page_content for t in texts], embeddings, index_name=index_name
    )

    docs = vectorstore.similarity_search(args.question)

    llm = ChatOpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    result = chain.invoke({"input_documents": docs, "question": args.question})
    print(result["output_text"])


if __name__ == "__main__":
    main()
