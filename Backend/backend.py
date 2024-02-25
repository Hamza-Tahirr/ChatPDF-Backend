# Required installations
# pip install langchain --upgrade
# !pip install pypdf

# Importing necessary libraries
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Loading environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YourAPIKey')

# Load data
loader = TextLoader(file_path="yahan File path dalna ./ something.pdf")
data = loader.load()

# Chunking data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(data)

# Create embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = Chroma.from_documents(texts, embeddings)

# Query documents
query = "What is Artificial Intelligence?"
docs = vectorstore.similarity_search(query)

# Initializing ChatOpenAI and running the chain
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")
chain.run(input_documents=docs, question=query)
