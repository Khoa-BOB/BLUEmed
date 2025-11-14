import os
import getpass

from typing_extensions import TypedDict
from IPython.display import Image, display

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from config.settings import settings
from app.core.prompts import EXPERT2_SYSTEM

print(EXPERT2_SYSTEM)

# # ----- Graph state -----


# # ----- LLM -----
# llm = ChatOllama(
#     model=settings.EXPERT_MODEL,
#     temperature=0.7,   # add randomness (change this to taste)
#     top_p=0.9,         # optional nucleus sampling
#     repeat_penalty=2.1 # optional, helps avoid repetition
# )

# # Graph state
# class State(TypedDict):
#     topic: str
#     joke: str
#     story: str
#     poem: str
#     combined_output: str


# # Nodes
# def call_llm_1(state: State):
#     """First LLM call to generate initial joke"""

#     msg = llm.invoke(f"Write a joke about {state['topic']}")
#     return {"joke": msg.content}


# def call_llm_2(state: State):
#     """Second LLM call to generate story"""

#     msg = llm.invoke(f"Write a story about {state['topic']}")
#     return {"story": msg.content}


# def call_llm_3(state: State):
#     """Third LLM call to generate poem"""

#     msg = llm.invoke(f"Write a poem about {state['topic']}")
#     return {"poem": msg.content}


# def aggregator(state: State):
#     """Combine the joke and story into a single output"""

#     combined = f"Here's a story, joke, and poem about {state['topic']}!\n\n"
#     combined += f"STORY:\n{state['story']}\n\n"
#     combined += f"JOKE:\n{state['joke']}\n\n"
#     combined += f"POEM:\n{state['poem']}"
#     return {"combined_output": combined}


# # Build workflow
# parallel_builder = StateGraph(State)

# # Add nodes
# parallel_builder.add_node("call_llm_1", call_llm_1)
# parallel_builder.add_node("call_llm_2", call_llm_2)
# parallel_builder.add_node("call_llm_3", call_llm_3)
# parallel_builder.add_node("aggregator", aggregator)

# # Add edges to connect nodes
# parallel_builder.add_edge(START, "call_llm_1")
# parallel_builder.add_edge(START, "call_llm_2")
# parallel_builder.add_edge(START, "call_llm_3")
# parallel_builder.add_edge("call_llm_1", "aggregator")
# parallel_builder.add_edge("call_llm_2", "aggregator")
# parallel_builder.add_edge("call_llm_3", "aggregator")
# parallel_builder.add_edge("aggregator", END)
# parallel_workflow = parallel_builder.compile()

# # Show workflow
# display(Image(parallel_workflow.get_graph().draw_mermaid_png()))

# # Invoke
# state = parallel_workflow.invoke({"topic": "cats"})
# print(state["combined_output"])


# from dotenv import load_dotenv
# import os
# import requests
# import json

# # Load the environment file
# load_dotenv()

# url = os.getenv("OLLAMA_URL")
# generate_endpoint = url + '/api/generate'

# print(generate_endpoint)

# data = {
#     "model" : "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M",
#     "prompt" : "what is the mascot of University of Memphis"
# }

# response = requests.post(generate_endpoint, json=data, stream = True)

# #Check the response status

# if response.status_code == 200:
#   print("Respnse: ", end=" ", flush=True)
#   # Iterate over the streaming response
#   for line in response.iter_lines():
#     if line:
#       #Decode the line and parse the JSON
#       decoded_line = line.decode("utf-8")
#       result = json.loads(decoded_line)

#       # Get the text from the response
#       generate_txt = result.get("response","")
#       print(generate_txt, end="", flush=True)
# else:
#   print("Error:", response.status_code, response.text)