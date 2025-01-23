import streamlit as st
from io import StringIO
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.memory import ConversationBufferMemory
import importlib.util
import nltk
import os
import fitz  # PyMuPDF for PDF parsing
import docx  # python-docx for Word files
import pandas as pd

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

  
# Check if libmagic is installed
libmagic_spec = importlib.util.find_spec("magic")
if libmagic_spec is None:
    st.error("libmagic is not installed. Please install it to proceed.")
else:
    import magic

gpt_model = "gpt-4o-mini"

# set local docs for langchain
chat_history = None
memory = None
loader = None
index = None 
retriever = None
llm = None
api_key = None 
upload = None

# path to database
DATA_DIR = "./database/"
os.makedirs(DATA_DIR, exist_ok=True)

# assistant prompt
pre_prompt = "You are a friendly and helpful teaching assistant called Cousin. You explain concepts in great depth using simple terms."

# titulo da pagina
st.markdown("<h1 style='text-align: center; color: white;'>Marta-GPT v1.0.1</h1>", unsafe_allow_html=True)


def extract_text_from_pdf(uploaded_file):
    """Extract text from an uploaded PDF file."""
    text = ""
    try:
        with fitz.open(stream=uploaded_file.getvalue(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    except Exception as e:
        st.error(f"❌ Error reading PDF: {e}")
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
        return extract_text_from_pdf(uploaded_file)
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


def setup_langchain():
    global chat_history, memory, loader, index, llm, retriever, api_key 

    chat_history = []
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

    if not api_key or api_key.strip() == "":
      st.error("⚠️ Please provide a valid OpenAI API key in the sidebar.")
      return
      
    # Set local docs for LangChain
    embeddings = OpenAIEmbeddings(api_key=api_key)
    loader = DirectoryLoader(DATA_DIR, glob="**/*.*")  # Load all file types

    docs = loader.load()  # Load the documents
    if not docs or len(docs) == 0:
        st.warning("No documents found. Upload files before starting.")
        return

    index = VectorstoreIndexCreator(vectorstore_cls=FAISS, embedding=embeddings).from_documents(docs)

    #set up chain params:
    llm = ChatOpenAI(model = gpt_model, api_key = api_key, temperature = 0.7, max_tokens = 128)
    retriever = index.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2, "score_threshold": 1, "fetch_k": 16})


def marta(question: str) -> str:
    # receives prompt from user, and returns answer
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
    )
    result = chain.invoke({"question": question, "chat_history": chat_history})
    answer = result.get('answer', 'Sorry, I could not find an answer.')
    chat_history.append((question, answer))

    return answer.lower()


# sidebar
with st.sidebar:
    
    st.sidebar.header("Provide a valid OpenAI API key 🗝")
    api_key = st.sidebar.text_input("Enter your API key:", type="password")
    
    if not api_key:
        st.warning("⚠️ Please enter a valid OpenAI API key to proceed.")
        st.stop()  # Stop execution until the user provides an API key
        
    st.header("Provide data files with relevant info 📄")
    upload = st.file_uploader("Upload a file", type=["pdf", "docx", "txt", "csv"])
    
    if upload is None:
        st.warning("⚠️ Please upload a valid file to proceed.")
        st.stop()  # Stop execution until the user uploads a file

    if upload:
        extracted_text = process_uploaded_file(upload)
        
        if extracted_text:
            file_path = os.path.join(DATA_DIR, upload.name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(extracted_text)

            st.success(f"✅ File '{upload.name}' uploaded successfully!")

            # Ensure setup_langchain is called after api_key is set
    setup_langchain()
        

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


prompt = st.chat_input()

if prompt is not None:

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # mandar a questao e receber resposta do langchain
    answer = marta(prompt)

    #mostrar resposta
    with st.chat_message("assistant"):
        st.write(answer)

    message = {"role": "assistant", "content": answer}
    st.session_state.messages.append(message)
