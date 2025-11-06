from dotenv import load_dotenv
import os
import requests
import json

# Load the environment file
load_dotenv()

url = os.getenv("OLLAMA_URL")
generate_endpoint = url + '/api/generate'

print(generate_endpoint)

data = {
    "model" : "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M",
    "prompt" : "what is the mascot of University of Memphis"
}

response = requests.post(generate_endpoint, json=data, stream = True)

#Check the response status

if response.status_code == 200:
  print("Respnse: ", end=" ", flush=True)
  # Iterate over the streaming response
  for line in response.iter_lines():
    if line:
      #Decode the line and parse the JSON
      decoded_line = line.decode("utf-8")
      result = json.loads(decoded_line)

      # Get the text from the response
      generate_txt = result.get("response","")
      print(generate_txt, end="", flush=True)
else:
  print("Error:", response.status_code, response.text)