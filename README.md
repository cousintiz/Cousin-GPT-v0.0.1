<h1 align="center">
    Marta-GPT v0.0.1
</h1>

A basic personal assistant/chatbot that can instantly answer your questions and complete tasks within your documents, It was developed using Large Language Models (LLMs), LangChain, and Retrieval-Augmented Generation (RAG).

##  Langchain Docs RAG Architecture:
![alt tag](pagev1.png)

## 💻Installation guide:
#### Clone the repository
```python
git@github.com:cousintiz/Marta-GPT-v0.0.1.git
```
#### If you have python3.10 intalled run this shell script otherwise follow the steps below:
```bash
./setup.sh
```
#### Create a virtual environment for the project's dependencies:
```python
python3 -m venv <environment-name>
```
#### Ativacte the virtual environment:
```python
<environment-name>\Scripts\Activate.ps1
```
#### Install the packages on the requirements.txt:
```python
pip3 install -r requirements.txt
```

## Usage
#### After all the steps above just run the App:
```bash
./app.sh
```
- Get an OpenAI API key [here](https://platform.openai.com/api-keys)
- Submit an text file document with the relevant information to feed into the model once you run the app.
#### This window window will be opened on your browser
![alt tag](pagev0.png)

## ⚒️Development Stack
- [Langchain](https://github.com/hwchase17/langchain)
- [Streamlit](https://streamlit.io/)
- [GPT 3.5-Turbo](https://platform.openai.com/docs/models/gpt-3-5-turbo)

## ⚙️Contribution
You are allowed use any part of the software or ideas found on this repository for anything, feel free to contribute, with improvements, issues or whatever you find useful...
