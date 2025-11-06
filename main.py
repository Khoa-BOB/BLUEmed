from ollama import chat
from ollama import ChatResponse


response: ChatResponse = chat(model='hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M', messages=[
  {
    'role': 'user',
    'content': 'What is the mascot of the University of Memphis?',
  },
])
print(response['message']['content'])
# or access fields directly from the response object
print(response.message.content)