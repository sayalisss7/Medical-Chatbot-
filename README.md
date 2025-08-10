# Medical Chatbot

A simple AI-powered medical chatbot built using **Streamlit**, **LangChain**, and **Hugging Face** models.  
It uses a CSV dataset of past responses to answer user medical queries in a similar style.

---

## 🚀 Features
- 💬 Takes user queries via a chat interface  
- 🔍 Searches a CSV dataset for similar past responses using **FAISS** vector search  
- 🤖 Generates advice using **Hugging Face** LLMs  
- 🔗 Uses **OpenAI API** (optional) for conversation handling  

---

## 📦 Requirements
Install dependencies from `requirments.txt`:

pip install -r requirments.txt


## 🔑 Environment Variables
Create a `.env` file in the project folder with:


HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_key



## 📂 File Structure
```plaintext
📦 Medical Chatbot
 ┣ 📜 app.py
 ┣ 📜 dataset.csv
 ┣ 📜 requirments.txt
 ┗ 📜 .env

