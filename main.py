import streamlit as st
from io import StringIO
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.prompts.prompt import PromptTemplate
import pickle
import importlib.util
import atexit
import nltk
from datetime import datetime
import pytz
from dotenv import load_dotenv
import requests
from datetime import datetime
import os
import fitz  # PyMuPDF for PDF parsing
import docx  # python-docx for Word files
import pandas as pd
from openai import OpenAI  # Updated import
import requests
import json
import time


nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Check if libmagic is installed
libmagic_spec = importlib.util.find_spec("magic")
if libmagic_spec is None:
    st.error("libmagic is not installed. Please install it to proceed.")
else:
    import magic

# set local docs for langchain
load_dotenv()
system = None # source of ans
chat_history = None
memory = None
loader = None
index = None 
retriever = None
llm = None
upload = None
title = "DocuChat AI"
gpt_model = "gpt-4o-mini"
ds_model = "deepseek-chat"
max_tokens = 256  # Add token limit
temperature = 0.7

# api keys
api_key = os.getenv('API_KEY')
deepseek_key = os.getenv('DEEPSEEK_API_KEY')
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

portugal_timezone = pytz.timezone('Europe/Lisbon')

# Get the current date in Portugal timezone
current_date_portugal = datetime.now(portugal_timezone).date()

# Formatting the date as 'YYYY-MM-DD'
current_date_str = current_date_portugal.strftime('%Y-%m-%d')

# url
deepseekUrl = "https://api.deepseek.com"

# words to filter
filter = ["i don't know.","to get real-time","don't have real-time", "not able to browse","unable to browse", "can’t browse", "sorry", "can't answer", "i don't have specific", "October 2023"]

# path to database
DATA_DIR = "./database/"
os.makedirs(DATA_DIR, exist_ok=True)

#Replace Docuchat AI Make.com webhook URL 
WEBHOOK_URL = "https://hook.eu2.make.com/2997z20s1mj8p4jylvw6ra6pzvwh3ho9"

# Set headers for JSON content
headers = {
    "Content-Type": "application/json",
    "User-Agent": "Python Webhook Client"
}

# We'll store chain & FAISS index in session_state to avoid re-initializing
if "chain" not in st.session_state:
    st.session_state.chain = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

# ------------------------------------
# Custom Prompts for Conversational RAG
# ------------------------------------
CONDENSE_QUESTION_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=(
        "You are a helpful Docuchat , an AI assistant that condenses the user's, Build by a company called DeepAgents, using deepseek question into a standalone question.\n\n"
        "Chat History:\n{chat_history}\n\n"
        "User's Last Question:\n{question}\n\n"
        "Please rewrite the last question into a self-contained question. "
        "Make it brief but clear."
    ),
)

COMBINE_DOCS_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are called DocuChat, a highly knowledgeable AI. Build by a company called DeepAgents,  Use the following document excerpts to craft your answer.\n"
        "If you cannot find the answer in the provided context, just say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Provide a detailed yet concise answer in simple terms:"
    ),
)

# assistant prompt
pre_prompt = "You are a DeepSeek-R1, a friendly and helpful teaching AI assistant. You explain concepts in great depth using simple terms."
ds_pre_prompt = "You are DeepSeek-R1, an AI friendly and helpful teaching assistant created exclusively by DeepSeek. if you dont know something just say i dont know"
# titulo da pagina
st.markdown(f"<h1 style='text-align: center; color: white;'>{title}</h1>", unsafe_allow_html=True)
temp_message = st.empty()

def extract_text_from_pdf(uploaded_file):
    """Extract text from an uploaded PDF file, handling corrupt files."""
    text = ""
    try:
        # Read file data
        file_bytes = uploaded_file.getvalue()
        
        # Validate if it's a real PDF
        if not file_bytes.startswith(b"%PDF"):
            st.error("❌ The uploaded file is not a valid PDF. Please upload a proper PDF document.")
            return None
        
        # Process the PDF
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text("text") + "\n"

        # Handle PDFs with only images (no text)
        if not text.strip():
            st.warning("⚠️ This PDF contains only images. OCR might be needed to extract text.")
        
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")
        return None
    
    return text


def extract_text_from_docx(docx_file):
    """Extract text from a Word document."""
    text = ""
    try:
        doc = docx.Document(docx_file)
        text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
    return text


def extract_text_from_csv(csv_file):
    """Extract text from a CSV file."""
    text = ""
    try:
        df = pd.read_csv(csv_file)
        text = df.to_string(index=False)  # Convert CSV content to text
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
    return text


def process_uploaded_file(uploaded_file):
    """Process different file types and extract text."""
    file_extension = uploaded_file.name.split(".")[-1].lower()

    if file_extension == "pdf":
        extracted_text = extract_text_from_pdf(uploaded_file)
        if not extracted_text:
            return None  # Skip if PDF processing fails
        return extracted_text
    elif file_extension == "docx":
        return extract_text_from_docx(uploaded_file)
    elif file_extension == "csv":
        return extract_text_from_csv(uploaded_file)
    elif file_extension == "txt":
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        return stringio.read()
    else:
        st.error(f"❌ Unsupported file format: {file_extension}")
        return None


def build_or_load_faiss_index(docs, embeddings, faiss_path: str):
    """
    Load FAISS index from disk if exists; otherwise build and save a new one.
    This avoids re-embedding large docs every session.
    """
    if os.path.exists(os.path.join(faiss_path, "index.faiss")):
        try:
            faiss_index = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
            return faiss_index
        except Exception as e:
           print(f"Error loading FAISS index: {e}")
    faiss_index = FAISS.from_documents(docs, embeddings)
    faiss_index.save_local(faiss_path)
    return faiss_index


def setup_langchain(filename):
    global chat_history, memory, loader, index, llm, retriever, api_key, max_tokens, gpt_model, temperature
    
    if filename is not None:

        chat_history = []
        memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

        if not api_key or api_key.strip() == "":
            st.error("⚠️ Please provide a valid OpenAI API key in the sidebar.")
            return
        
        # Set local docs for LangChain
        embeddings = OpenAIEmbeddings(api_key=api_key)


        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path):
            st.error(f"❌ File '{filename}' not found.")
            return
        #loader = DirectoryLoader(DATA_DIR, glob="**/*.*")  # Load all file types

        loader = TextLoader(file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = loader.load_and_split(text_splitter)
        #docs = loader.load()
        if not docs or len(docs) == 0:
            st.warning("No documents found. Upload files before starting.")
            return
        
        # Initialize LangChain
        #index = VectorstoreIndexCreator(vectorstore_cls=FAISS, embedding=embeddings).from_documents(docs)
        
        faiss_index_path = os.path.join(DATA_DIR, "faiss_index")
    # Build or load FAISS Index
        st.session_state.faiss_index = build_or_load_faiss_index(docs, embeddings, faiss_index_path)

            
        # llm
        llm = ChatOpenAI(
            model=gpt_model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
        #retriever = index.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2, "score_threshold": 1, "fetch_k": 16})
        retriever = st.session_state.faiss_index.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,  # Retrieve more results for improved context
                "lambda_mult": 0.5,  # Balance diversity & relevance in MMR
                "fetch_k": 20,  # Fetch more documents internally for filtering
            })
        

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            combine_docs_chain_kwargs={"prompt": COMBINE_DOCS_PROMPT},
            memory=st.session_state.memory
        )
        return chain
    else:
        return


def search_web(query):
    """Use Google Search API (SerpAPI) to fetch relevant search results."""
    print("🔍 Searching the web...")

    if not SERPAPI_KEY:
        return "⚠️ No search API key found. Please configure SerpAPI."

    try:
        search_url = "https://serpapi.com/search"
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_KEY
        }
        response = requests.get(search_url, params=params)
        data = response.json()

        # Extract relevant snippets
        if "organic_results" in data:
            first_result = data["organic_results"][0]  # Get the first search result
            title = first_result.get("title", "No Title")
            snippet = first_result.get("snippet", "No Description Available")
            link = first_result.get("link", "")

            return f"🔎 **{title}**\n\n{snippet}\n\n[Read more]({link})"

        return "⚠️ No relevant information found online."
    
    except Exception as e:
        return f"⚠️ Web search error: {e}"


def gpt(prompt) -> str:
    print("running gpt...")
    try:
        client = OpenAI(api_key=deepseek_key, base_url=deepseekUrl)
        response = client.chat.completions.create(
            model=ds_model,
            messages=[
                {"role": "system", "content": ds_pre_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens  # Apply token limit
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        st.error(f"⚠️ OpenAI API error: {e}")
        return ""


def marta(prompt: str) -> str:
    # receives prompt from user, and returns answer
    print("runnig marta...")

    chain = st.session_state.chain
    if not chain:
        return None

    # RAG from local docs
    result = chain({"question": prompt})
    answer = result.get("answer", "").strip()

    result = chain({"question": prompt})
    answer = result.get("answer", "").strip()
    print(answer)
    chat_history.append((prompt, answer))

    return answer


def agent_run(prompt: str) -> str:
    global temp_message, source
    st.session_state.id = str(time.time()) # Generate unique session ID

    answer = marta(prompt)
    source = "RAG"

    if not answer or any(phrase in answer.lower() for phrase in filter):
        temp_message.empty()
        temp_message.markdown("🤔 Thinking... Fetching deeper insights...")
        
        # ✅ Call OpenAI only if FAISS retrieval fails
        answer = gpt(prompt)
        source = "GPT"

    temp_message.empty()

    if not answer or any(phrase in answer.lower() for phrase in filter):
        temp_message.markdown("🌐 Searching the web...")
        answer = search_web(prompt)
        source = "WEB"

    temp_message.empty()
    return answer


def update_usage(source, prompt):
    """Update usage stats for the current session."""
    
    print("session id:",current_date_str)
    payload = {
        "data": {
            "prompt": prompt,
            "source": source,
            "date":current_date_str
        }
    }

    try:
        response = requests.post(
                    WEBHOOK_URL,
                    data=json.dumps(payload),
                    headers=headers
        )
        if response.status_code in [200, 201]:
            print("Usage updated successfully!")
    except requests.exceptions.RequestException as e:
        print(f"Failed to update usage: {str(e)}")


def cleanup_files():
    """Delete all uploaded files when the session ends."""
    for file in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file)
        os.remove(file_path)
    print("🧹 All uploaded files have been deleted.")

# ✅ Register cleanup function to run when script stops
atexit.register(cleanup_files)

# sidebar
with st.sidebar:
        
    st.markdown("<h3 style='text-align: center; color: white;'>Provide files with relevant info 📄</h3>", unsafe_allow_html=True)
    upload = st.file_uploader("Upload a file", type=["pdf", "docx", "txt", "csv"])
    
    if upload is None:
        st.write("...")
        #st.stop()  # Stop execution until the user uploads a file

    if upload:
        extracted_text = process_uploaded_file(upload)
        fname = upload.name
        if extracted_text:
            file_path = os.path.join(DATA_DIR, upload.name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(extracted_text)

            with st.spinner("Processing documents..."):
                # Ensure  Fsetup_langchain is called after api_key is set
                st.session_state.chain = setup_langchain(fname)
                st.success(f"✅ File uploaded successfully!")


st.sidebar.markdown("<h3 style='text-align: center; color: white;'>Configure Model's Performance</h3>", unsafe_allow_html=True)
temperature=st.sidebar.slider("Temperature", 0.0, 1.0, 0.7),
max_tokens=st.sidebar.number_input("Max Tokens", min_value=128, max_value=512, value=256)
       
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input()

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Create a temporary message container
    temp_message = st.empty()
    temp_message.markdown("🤔 Thinking... Please wait...")

    # mandar a questao e receber resposta do langchain
    answer = agent_run(prompt)

    temp_message.empty() # clear temp message
    #mostrar resposta
    with st.chat_message("assistant"):
        st.write(answer)

    # Update usage stats
    update_usage(source, prompt) # send prompt and source to webhook

    # Add assistant message to the session   
    st.session_state.messages.append({"role": "assistant", "content": answer})
