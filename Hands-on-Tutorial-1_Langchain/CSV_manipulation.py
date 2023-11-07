import streamlit as st
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI

from dotenv import load_dotenv
load_dotenv()

def query_agent(data, query):
	df = pd.read_csv(data)
	agent = create_pandas_dataframe_agent(OpenAI(temperature = 0.2), df, verbose = True)
	response = agent.run("First, import any python libraries that you'll need. Add a hashtag # to the end of every line of python code." + str(query))
	return response

st.header("Upload your spreadsheet here:")
data = st.file_uploader("Upload CSV file", type = "csv")
query = st.text_area("What can I help you with?")
button = st.button("Submit")

if button:
	response =  query_agent(data,query)
	st.write(response)
