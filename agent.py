from langchain_google_vertexai import ChatVertexAI
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv

import os
load_dotenv()
model = ChatVertexAI(model="gemini-1.5-flash")
search = TavilySearchResults(max_results=2)

tools = [search]

# model_with_tools = model.bind_tools(tools)

# response = model_with_tools.invoke([HumanMessage(content="What is the weather in Istanbul/Maltepe")])

# print(f"ContentString: {response.content}")
# print(f"ToolCalls: {response.tool_calls}")

agent_executor = create_react_agent(model, tools)

response = agent_executor.invoke(
    {"messages": [HumanMessage(content="whats the weather in sf?")]}
)

print(response["messages"])