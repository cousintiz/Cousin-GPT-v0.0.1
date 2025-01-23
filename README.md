<h1 align="center">
    Marta-GPT v1.0.1
</h1>

A basic personal assistant/chatbot that can instantly answer your questions and complete tasks using your documents. Developed using Large Language Models (LLMs), LangChain, and Retrieval-Augmented Generation (RAG).

🚀 **Live App:** [Marta-GPT](https://marta-gpt.streamlit.app/)

## 📖 Langchain Docs RAG Architecture:
![alt tag](pagev1.png)

---
## 💻 Installation Guide (Run Locally)

### **1️⃣ Clone the repository**
```bash
git clone git@github.com:cousintiz/Marta-GPT-v0.0.1.git
cd Marta-GPT-v0.0.1
```

### **2️⃣ Create a virtual environment**
```bash
python3 -m venv venv
```

### **3️⃣ Activate the virtual environment**
#### **MacOS/Linux:**
```bash
source venv/bin/activate
```
#### **Windows (Command Prompt):**
```cmd
venv\Scripts\activate
```
#### **Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

### **4️⃣ Install dependencies**
```bash
pip install -r requirements.txt
```

### **5️⃣ Run the App Locally**
```bash
streamlit run main.py
```

### **6️⃣ Open the app in your browser**
Once the app starts, it will automatically open in your default browser. If not, go to:
```
http://localhost:8501
```

### **7️⃣ Get an OpenAI API Key**
- Create an OpenAI account if you don’t have one.
- Get your API key [here](https://platform.openai.com/api-keys)
- Enter the API key in the app when prompted.

### **8️⃣ Upload Your Documents**
- Submit a `.txt` file with relevant information to feed into the model.
- Marta-GPT will process the file and provide intelligent answers based on your document content.

## 🌐 Running on The App Cloud
Marta-GPT is hosted online at:
🔗 **[Marta-GPT](https://marta-gpt.streamlit.app/)**

No installation is required! Simply visit the link, enter your OpenAI API key, and upload your document.

---
## ⚒️ Development Stack
- [Langchain](https://github.com/hwchase17/langchain)
- [Streamlit](https://streamlit.io/)
- [GPT-4o](https://platform.openai.com/docs/models/gpt-4o)
- [FAISS](https://github.com/facebookresearch/faiss) (Vector Search Engine)

---
## 🛠️ Contribution
This project is open-source! Feel free to:
- Submit pull requests with improvements 🛠️
- Report bugs or issues 🐛
- Suggest new features 🚀

Any contributions are welcome! 😊

