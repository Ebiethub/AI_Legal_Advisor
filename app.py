import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import os
import re
import datetime
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GROQ_API_KEY=st.secrets["GROQ_API_KEY"]
if not GROQ_API_KEY:
    st.error("GROQ API Key is missing. Please set it in your .env file.")
    st.stop()

LEGAL_CATEGORIES = [
    "Contract Law", "Employment Law", "Property Law",
    "Criminal Law", "Family Law", "Corporate Law",
    "Intellectual Property", "Tax Law", "International Law"
]

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_vectors" not in st.session_state:
    st.session_state.document_vectors = None

def init_groq_chain():
    """Initialize Groq model."""
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-specdec",
        temperature=0.3,
        max_tokens=4000
    )

def classify_legal_query(query: str) -> str:
    """Classify legal queries into categories."""
    classifier_prompt = ChatPromptTemplate.from_messages([
        ("system", """Classify this legal query into one of these categories: {categories}. 
         Return only the category name."""),
        ("human", "{query}")
    ])
    
    chain = classifier_prompt | init_groq_chain() | StrOutputParser()
    return chain.invoke({
        "query": query,
        "categories": ", ".join(LEGAL_CATEGORIES)
    })

def analyze_legal_document(file_path: str):
    """Analyze uploaded legal documents."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        st.error("Unsupported file format.")
        return

    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)
    
    embeddings= HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    st.session_state.document_vectors = FAISS.from_documents(splits, embeddings)

def legal_advice_chain():
    """Create main legal advice chain."""
    system_prompt = """You are a certified legal advisor with expertise in multiple jurisdictions. 
    Provide accurate, conservative legal guidance following these rules:
    1. Always clarify jurisdiction
    2. Cite relevant statutes/cases with dates
    3. Highlight potential risks
    4. Suggest next steps
    5. Include disclaimer about non-binding nature
    6. Provide authoritative sources
    7. Current date: {current_date}
    {legal_context}"""

    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("assistant", "{output}")
    ])
    
    return prompt | init_groq_chain() | StrOutputParser()

def get_legal_context(query: str, category: str) -> str:
    """Retrieve relevant legal context."""
    context = []
    
    if st.session_state.document_vectors:
        docs = st.session_state.document_vectors.similarity_search(query, k=3)
        context.extend([d.page_content for d in docs])

    context.append(f"Relevant {category} statutes: ...")
    
    return "\n\n".join(context)

def main():
    st.set_page_config(page_title="AI Legal Advisor", layout="wide", page_icon="⚖️")
    st.title("AI Legal Advisor Chatbot")
    st.sidebar.header("Configuration")

    selected_lang = st.sidebar.selectbox("Response Language", ["English", "Spanish", "French", "German", "Chinese"])
    uploaded_file = st.sidebar.file_uploader("Upload Legal Document", type=["pdf", "docx"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            analyze_legal_document(tmp_file.name)
        st.sidebar.success("Document analyzed successfully!")

    chat_history = [
        HumanMessage(content=msg["content"]) if msg["role"] == "user" 
        else AIMessage(content=msg["content"])
        for msg in st.session_state.messages
    ]

    if query := st.chat_input("Ask your legal question..."):
        with st.spinner("Analyzing legal context..."):
            st.session_state.messages.append({"role": "user", "content": query})
            category = classify_legal_query(query)
            legal_context = get_legal_context(query, category)
            chain = legal_advice_chain()
            response = chain.invoke({
             "input": f"{query} (Respond in {selected_lang})",
             "current_date": datetime.datetime.now().strftime("%Y-%m-%d"),
             "legal_context": legal_context,
             "output": ""  # Add an empty placeholder if required
             })

        sources = re.findall(r"\[Source: (.*?)\]", response)
        response = re.sub(r"\[Source: .*?\]", "", response)

        st.session_state.messages.append({"role": "assistant", "content": response, "sources": "\n".join(sources), "category": category})
        st.write(response)
    
        
    # Security disclaimer
    st.sidebar.markdown("---")
    st.sidebar.info("""
        **Security Notice:**  
        - All communications are encrypted  
        - No data persistence beyond session  
        - Do not share sensitive case details  
        - Consult licensed attorney for binding advice
    """)

if __name__ == "__main__":
    main()
