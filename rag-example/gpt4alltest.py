from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.llms import GPT4All
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain


# Load and split an example document.
# We'll use a blog post on agents as an example.

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Next, the below steps will download the HuggingFaceEmbeddings embeddings locally (if you don't already have them).

vectorstore = Chroma.from_documents(documents=all_splits, embedding=HuggingFaceEmbeddings())


# Loading the GPT4All LLM model

llm = GPT4All(
    model="C:/Users/willi/AppData/Local/nomic.ai/GPT4All/gpt4all-falcon-newbpe-q4_0.gguf",
    max_tokens=2048,
)

# Integrating the documents from our vector database with a LLM model through the ConversationalRetrievalChain.

conversation = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    verbose=True,
)

chat_history = []
question = "What are the approaches to Task Decomposition?"
response = conversation({"question": question, "chat_history": chat_history})
answer = response['answer']

print(f"Got response from llm: {answer}")