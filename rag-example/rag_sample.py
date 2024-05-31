from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.llms import GPT4All
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load and split an example document.
# We'll use a blog post on agents as an example.

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Next, the below steps will download the HuggingFaceEmbeddings embeddings locally (if you don't already have them).

vectorstore = Chroma.from_documents(documents=all_splits, embedding=HuggingFaceEmbeddings())

# Test similarity search is working with our local embeddings.
print("-----------------------------")
print("Question: What are the approaches to Task Decomposition?")
question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)
print("-----------------------------")
print(f"Number of docs found: {len(docs)}")
print("\nSample doc:\n")
print("-----------------------------")
print(docs[0])
print("-----------------------------")



