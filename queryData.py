from typing_extensions import TypedDict,List
from langchain_core.documents import Document
from langchain import hub
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import START, StateGraph
from main import vector_store

prompt = hub.pull("rlm/rag-prompt")

# You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't     know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
# Question: {question} 
# Context: {context} 
# Answer:

llm = ChatVertexAI(model="gemini-1.5-flash")


class State(TypedDict) : 
    question:str
    context:List[Document]
    answer:str

def retrieve(state:State) : 
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context":retrieved_docs}

def generate(state:State) :
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question":state["question"],"context":docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve,generate])
graph_builder.add_edge(START,"retrieve")
graph = graph_builder.compile()

# response = graph.invoke({"question": "Benim emailim nedir"})
# print(response["answer"])


