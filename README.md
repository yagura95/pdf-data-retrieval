# Description
This is a RAG(Retrieval-Augumentaded Generation) model that can be used to analyse pdf documents contents.

# Install 
## Ollama and local models
It uses Mistral model wiht Ollama(local AI model managment). You can install Ollama [here](https://ollama.com/download). After installing Ollama, you can pull the required model to use it locally.
```
ollama pull mistral
```

If you don't have Ollama starting at boot, you will need to manually start it.
```
ollama serve &
```

## Python requirements
Create a virtual environment with venv
```
python3 -m venv /path/to/new/virtual/environment
cd /path/to/new/virtual/environment
source env/bin/activate
```

Install dependencies
```
pip install -r requirements.txt
```

# Usage 
```
python3 main.py <file_path>
```






