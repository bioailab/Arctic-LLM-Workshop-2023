from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredPDFLoader
import uuid
from langchain.schema import Document

# Initialize the LLM API Key(s)
from dotenv import load_dotenv
load_dotenv()

# Load documents

loader = UnstructuredPDFLoader("thesis.pdf")
text = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
	chunk_size = 500,
	chunk_overlap  = 20,
	length_function = len,
	add_start_index = True,
)
chunks = text_splitter.split_documents([text[0]])

print("Total number of split document chunks: " + str(len(chunks)))

#Choose which LLM Provider

which_model = "openai"
#which_model = "huggingface"

features = 0
if(which_model == "openai"):
	features = 1536
if(which_model == "huggingface"):
	features = 384

#Bring in the LLM

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

embedding_function = None
if(which_model == "openai"):
	embedding_function = OpenAIEmbeddings()
if(which_model == "huggingface"):
	embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.llms import OpenAI

llm = None
if(which_model == "openai"):
	llm = OpenAI(model_name="text-davinci-003",temperature=0)
if(which_model == "huggingface"):
	llm=HuggingFaceHub(repo_id="bigscience/bloom")

def ask_the_model(user_input):
	
	# Create Chroma Vectorstore database 
	
	db = Chroma()
	if(db._collection.count() == 0):
		print("Initialized vectorstore with related texts")
		db = Chroma.from_documents(chunks, embedding_function)
	else:
		empty_document = Document(page_content="...", metadata={'source': 'this_doesnt_exist.pdf'})
		empty_chunks = text_splitter.split_documents([empty_document])
		db = Chroma.from_documents(empty_chunks, embedding_function)
		print("Loaded previous vectorstore")
	
	# Get chunks of the document which are similar to the query
	
	similar_chunks = db.similarity_search(user_input, 15)
	for chunk in similar_chunks:
		print(" ")
		print("relevant doc: " + str(chunk))
	print("number of documents in db: " + str(len(db.get()['documents'])))
	
	# Send everything to the LLM
	
	query = "You are a chatbot having a conversation with a human. Given the following extracted parts of a long document and a question, create a final answer.\n"
	query = query + str(similar_chunks) + "\n"
	query = query + "This is the end of the extracted parts of a long document. The question that you are meant to answer is the following:\n"
	query = query + str(user_input)
	response = llm(query)
	return response

import streamlit as st

st.header("Query your internal documents")
user_input = st.text_area("Ask me a question about the attached documents")
button = st.button("Submit")

if button:
	response = ask_the_model(user_input)
	st.write(response)
