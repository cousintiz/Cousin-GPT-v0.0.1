### **🚀 DocuChat AI - Project Structure & Features**  
**DocuChat AI** is an advanced **document-based chatbot** that enables users to **upload files (PDF, DOCX, TXT, CSV)** and ask questions based on their content. It combines **LangChain for retrieval, OpenAI’s GPT for responses, and web search (Google/Bing) for real-time data**.

This guide provides a **detailed breakdown** of the files in the repository, explaining their purpose and how they contribute to the app’s functionality.

---

## **📂 Project Directory & File Breakdown**
Below is a detailed **overview of each file and folder** in the project:

### **🔹 `.github/workflows/`** *(GitHub Actions - CI/CD Automation)*
- This directory contains **workflow files** for GitHub Actions.
- Typically used to **automate testing, deployment, and version control** of the application.

---

### **🔹 `.env`** *(Environment Variables - 🔑 API Keys & Config)*
- Stores **sensitive credentials** such as:
  - **OpenAI API Key** (for GPT model responses)
  - **SerpAPI Key** (for Google Search integration)
  - **Bing API Key** (alternative web search)
- Example `.env` file:
  ```ini
  API_KEY=your_openai_api_key
  SERPAPI_KEY=your_serpapi_key
  BING_API_KEY=your_bing_api_key
  ```

💡 **Ensure this file is listed in `.gitignore` to prevent exposing credentials!**

---

### **🔹 `.gitignore`** *(Git Ignore Rules - 🚫 Prevents Unwanted Files in Git)*
- Prevents **unnecessary or sensitive files** from being committed to Git.
- Common entries include:
  ```gitignore
  .env
  __pycache__/
  database/
  venv/
  ```

---

### **🔹 `LICENSE`** *(Open Source License - 📜 Project Rights & Permissions)*
- Defines the **terms and conditions** for using, modifying, and distributing the software.
- This project is likely **MIT licensed**, allowing free usage with attribution.

---

### **🔹 `README.md`** *(Documentation - 📄 How to Install & Use the App)*
- **Main guide** for users & developers.
- Contains:
  ✅ **Installation instructions**  
  ✅ **How to use the chatbot**  
  ✅ **API setup guide**  
  ✅ **Features & tech stack**  
  ✅ **How to contribute**  
  ✅ **Live demo link (if hosted online)**  

---

### **🔹 `app.sh`** *(Shell Script - 🔧 Quick Setup & Deployment)*
- A **bash script** to automate setting up the environment.
- Typically used for:
  ✅ **Activating virtual environments**  
  ✅ **Installing dependencies**  
  ✅ **Starting the application**  

---

### **🔹 `main.py`** *(Core Python Script - 🧠 AI Chatbot Implementation)*
This is the **heart of the DocuChat AI** system. It powers:
✅ **File Upload Handling** (PDF, DOCX, TXT, CSV)  
✅ **LangChain + FAISS** for document retrieval  
✅ **GPT-4o** for intelligent responses  
✅ **Web Search** (Google/Bing) fallback if GPT doesn’t know  
✅ **Automatic File Cleanup** (Deletes files after session)  

---

### **🔹 `pagev0.png` & `pagev1.png`** *(App UI Mockups - 🎨 Visual Guide)*
- These **images** likely contain **screenshots or architecture diagrams**.
- Used for **README documentation** or **UI previews**.

---

### **🔹 `requirements.txt`** *(Dependency Management - 📦 Required Python Packages)*
- Lists **all Python libraries** required for the chatbot.
- Install dependencies using:
  ```bash
  pip install -r requirements.txt
  ```
- Likely includes:
  ```ini
  streamlit
  langchain
  openai
  faiss-cpu
  pypdf
  python-docx
  pandas
  requests
  dotenv
  ```
- **Ensures that all required packages** are installed for a **smooth setup**.

---

### **🔹 `setup.sh`** *(Shell Script - 🛠️ Auto Setup for Deployment)*
- **Automates environment setup** for **new installations**.
- Common operations:
  ✅ **Create virtual environment**  
  ✅ **Install dependencies**  
  ✅ **Setup API keys**  
  ✅ **Launch application**  

---

## **🚀 How the App Works**
1️⃣ **User Uploads a Document**  
   - Supported formats: **PDF, DOCX, TXT, CSV**  
   - File is saved **temporarily** in `database/`  

2️⃣ **AI Reads the Document**  
   - Uses **LangChain + FAISS** to process the content  
   - Creates a **vector store** for efficient retrieval  

3️⃣ **User Asks a Question**  
   - First, the system tries to **retrieve** an answer from the document  
   - If no answer is found, **GPT-4o** generates a response  
   - If GPT also doesn’t know, it **searches the web**  

4️⃣ **AI Responds & Session Ends**  
   - AI **answers the user’s question**  
   - Once the session ends, all **uploaded files are deleted** for security  

---

## **🌐 Running on the Cloud**
**DocuChat AI is also hosted online!**  
🔗 **[Try it here](https://marta-gpt.streamlit.app/)**  

✅ **No installation needed**  
✅ **Just upload a document & start asking questions!**  

---

## **⚒️ Technology Stack**
| Component | Technology |
|-----------|------------|
| **LLM** | GPT-4o (via OpenAI API) |
| **Vector Search** | FAISS |
| **UI** | Streamlit |
| **Document Processing** | PyMuPDF (PDF), python-docx (DOCX), Pandas (CSV) |
| **AI Framework** | LangChain |
| **Web Search** | SerpAPI (Google), Bing Search API |

---

## **🛠️ Contribution**
This project is **open-source!** 🎉  
**Ways to Contribute:**  
✅ Improve **document parsing**  
✅ Add **multi-language support**  
✅ Optimize **search algorithms**  
✅ Fix **bugs & issues**  
✅ Suggest **new features** 🚀  

---

## **📜 License**
This project is licensed under the **MIT License**, meaning it’s free to use, modify, and share with attribution.  

---
### **🔥 Ready to Try?**
1️⃣ Clone the repo  
2️⃣ Install dependencies  
3️⃣ Set up your API keys  
4️⃣ Start **chatting with your documents!** 🚀  

---
**📌 Questions? Need help?** Contact us on **GitHub Discussions**! 😊
