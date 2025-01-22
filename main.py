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

# path to database
infofile = "./database/data.txt"  

# assistant prompt
pre_prompt = "You are a friendly and helpful teaching assistant called Cousin. You explain concepts in great depth using simple terms."

# titulo da pagina
st.markdown("<h1 style='text-align: center; color: white;'>Marta-GPT v0.0.1</h1>", unsafe_allow_html=True)


def setup_langchain():
    global chat_history, memory, loader, index, llm, retriever, api_key 

    chat_history = []
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

    # set local docs for langchain
    embeddings = OpenAIEmbeddings(api_key = api_key)
    loader = DirectoryLoader("database/", glob= "**/*.txt")
    index = VectorstoreIndexCreator(vectorstore_cls=FAISS,embedding = embeddings).from_loaders([loader])

    #set up chain params:
    llm = ChatOpenAI(model = gpt_model, api_key = api_key, temperature = 1, max_tokens = 128)
    retriever = index.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2, "score_threshold": 1, "fetch_k": 16})


def save_data(data: str) -> None:
    """
    Receives User data, to be embedded to the model
    """
    try:
        with open(infofile, 'a') as file:
            file.write(data)
        print("Successfully saved data.")
    except Exception as e:
        print(f"Error {e} has occurred")


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
    
    st.header("Provide a valid OpenAI API key🗝")
    
    while api_key is None:
        api_key = st.sidebar.text_input("your key:", type="password")

    if api_key:
        st.header("Provide data files with relevant info📄")
        upload = st.file_uploader("Upload a txt file")
        
        if upload is not None:
            stringio = StringIO(upload.getvalue().decode("utf-8"))
            datafile = stringio.read()
            save_data(datafile) # save data from file in path database

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


