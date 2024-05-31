from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.llms import GPT4All
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain

# Load and split an example document.
DB_PATH = "chroma_db"

loader = CSVLoader(file_path="imdb_top_1000.csv", encoding="utf-8")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Next, the below steps will download the HuggingFaceEmbeddings embeddings locally (if you don't already have them).

# transformer model https://huggingface.co/sentence-transformers/all-mpnet-base-v2/tree/main
enbeddings_model = HuggingFaceEmbeddings(model_name="C:/aulas/ia/transformers/all-mpnet-base-v2")

vectorstore = Chroma.from_documents(documents=all_splits, embedding=enbeddings_model, persist_directory=DB_PATH)

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
    verbose=False,
)

chat_history = []
while (True):
    question = input("\nDigite a sua pergunta sobre filmes: ")
    response = conversation.invoke({"question": question, "chat_history": chat_history})
    answer = response['answer']

    print(f"\nResposta: {answer}")
    if input("\nContinuar?[s/n]") != "s":
        break
