# ChatPDF Backend

A command-line Python script that answers questions about a PDF. It stores the document's text as OpenAI embeddings in a Pinecone index, finds the passages most relevant to your question and asks an OpenAI chat model to answer from them.

## Features

- Loads a PDF with PyMuPDF and splits it into 500-character chunks with 50 characters of overlap
- Embeds the chunks with OpenAI embeddings and stores them in a Pinecone index
- Retrieves the 4 most similar chunks for the question
- Answers with a LangChain "stuff" question-answering chain using `ChatOpenAI` at temperature 0
- Reads API keys from environment variables or a `.env` file

## Tech stack

- Python
- LangChain (`langchain`, `langchain-community`, `langchain-openai`, `langchain-pinecone`)
- OpenAI API (embeddings and chat model)
- Pinecone vector database
- PyMuPDF
- python-dotenv

## Project structure

```
ChatPDF-Backend/
├── Backend/
│   ├── backend.py      # the script
│   └── .env.example    # template for the environment variables
├── requirements.txt
├── README.md
└── LICENSE
```

## Setup

You need Python 3.10 to 3.13, an OpenAI API key and a Pinecone account.

1. Clone the repository and install the dependencies:

   ```bash
   git clone https://github.com/Hamza-Tahirr/ChatPDF-Backend.git
   cd ChatPDF-Backend
   python -m venv venv
   source venv/bin/activate        # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. In the Pinecone console, create an index named `chatpdf` with **1536** dimensions (the size of OpenAI's `text-embedding-ada-002` embeddings, which `OpenAIEmbeddings` uses by default) and the **cosine** metric.

3. Copy the environment template and fill in your keys:

   ```bash
   cp Backend/.env.example Backend/.env
   ```

   | Variable | Description |
   | --- | --- |
   | `OPENAI_API_KEY` | OpenAI API key |
   | `PINECONE_API_KEY` | Pinecone API key |
   | `PINECONE_INDEX_NAME` | Name of the Pinecone index (optional, defaults to `chatpdf`) |

## Usage

Pass the path to a PDF and a question:

```bash
python Backend/backend.py path/to/document.pdf "What is this document about?"
```

The answer is printed to the terminal.

Every run embeds the PDF and uploads its chunks to the index again, so running the same file several times stores duplicate vectors. Delete the vectors in the Pinecone console if you want to start fresh.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
