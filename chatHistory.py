# this page is not connected with main.py and queryData.py
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage,trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

model = ChatVertexAI(model="gemini-1.5-flash")
# ----------- without langgraph

# response = model.invoke(
#     [
#         HumanMessage(content="Hi! I'm Bob"),
#         AIMessage(content="Hello Bob! How can I assist you today?"),
#         HumanMessage(content="What's my name?"),
#     ]
# )
# print(response.content)

# ----------- with langgraph without managing chat history

# workflow = StateGraph(state_schema=MessagesState)

# def call_llm(state:MessagesState) : 
#     response = model.invoke(state["messages"])
#     return {"messages":response}


# workflow.add_edge(START,"model")
# workflow.add_node("model",call_llm)

# memory = MemorySaver()
# app = workflow.compile(checkpointer=memory)

# config = {"configurable": {"thread_id": "abc123"}}

# output = app.invoke({"messages":HumanMessage("Hi I'm Emirhan.")},config)
# output2 = app.invoke({"messages":HumanMessage("What is my name")},config)

# output2["messages"][-1].pretty_print()

# ----------- with langgraph and manage chat histor with trimmer

## one important think is managing conversation history

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)
messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]
res = trimmer.invoke(messages)

print(res)

