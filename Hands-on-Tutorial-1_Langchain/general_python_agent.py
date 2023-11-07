from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.llms.openai import OpenAI
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
load_dotenv()

query = input("What can I help you with?  ")

agent = create_python_agent(
	OpenAI(temperature=0.5), 
	tool=PythonREPLTool(), 
	verbose=True,
	agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
	agent_executor_kwargs={"handle_parsing_errors": True},
)

print(agent.run("This is a linux machine. First, import any python libraries that you'll need. " + str(query) + ". If you get the same error twice, try a different method. Before you finish, please also print a summary of any results to the console with a print statement. Add a hashtag # to the end of python code."))
