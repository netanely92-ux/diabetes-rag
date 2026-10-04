import streamlit as st
import csv
import os
import time
import uuid
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

DB_DIR = "./chroma_db"
LOG_FILE = "chat_interactions.csv"
GROQ_API_KEY = "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"

st.set_page_config(
    page_title="עוזר סוכרת קליני",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# עיצוב מודרני מיושר לימין (RTL)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@300;400;500;600&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Rubik', sans-serif !important;
        background-color: #f7f9fc !important;
        direction: rtl !important;
        text-align: right !important;
    }

    .hero-container {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.75rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(37, 99, 235, 0.15);
    }
    .hero-container h2 {
        margin: 0;
        font-weight: 600;
        font-size: 1.6rem;
        color: #ffffff;
    }
    .hero-container p {
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
        opacity: 0.9;
    }

    div[data-testid="stChatMessage"] {
        direction: rtl !important;
        text-align: right !important;
        padding: 1rem 1.25rem !important;
        border-radius: 16px !important;
        margin-bottom: 0.75rem !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }

    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background-color: #e0e7ff !important;
        border: 1px solid #c7d2fe !important;
    }

    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-right: 4px solid #2563eb !important;
    }

    ul, ol {
        direction: rtl !important;
        text-align: right !important;
        padding-right: 1.4rem !important;
        padding-left: 0 !important;
        margin: 0.5rem 0 !important;
    }
    li {
        margin-bottom: 0.25rem !important;
    }

    div[data-testid="stChatInput"] textarea {
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Rubik', sans-serif !important;
        border-radius: 12px !important;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-container">
    <h2>🩺 עוזר מידע קליני בנושא סוכרת</h2>
    <p>מענה מהיר לשאלות בנושאי מניעה, תסמינים, תזונה וטיפול קליני</p>
</div>
""", unsafe_allow_html=True)

def log_interaction(session_id: str, question: str, answer: str, sources: list, latency: float):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Session_ID",
                "User_Question",
                "Model_Answer",
                "Retrieved_Sources",
                "Latency_Seconds"
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            session_id,
            question,
            answer,
            "; ".join(sources),
            round(latency, 2)
        ])

@st.cache_resource
def load_rag():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # שימוש ב-llama-3.1-8b-instant למהירות מקסימלית ואמינות API
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=GROQ_API_KEY,
        temperature=0.3
    )
    return retriever, llm

retriever, llm = load_rag()

SYSTEM_PROMPT = """אתה עוזר וירטואלי חכם, אדיב, רהוט ונעים לשיחה, המתמחה במידע קליני בנושא סוכרת.

קטעי המידע הבאים עומדים לרשותך מהמאגר:
{context}

הנחיות לתגובה:
1. שיחה כללית וברכות: אם המשתמש פותח בברכה ("שלום", "היי", "מה נשמע", "בוקר טוב") או שואל שאלה כללית על מי אתה, הגב בצורה טבעית, חמה ואנושית. ספר בקצרה שאתה כאן כדי לסייע בכל שאלה בנושא סוכרת, מניעה, תזונה ובריאות.
2. שאלות מקצועיות על סוכרת: נסח תשובה ברורה, ממוקדת ומקצועית בהתבסס על המידע הרפואי. השתמש בפסקאות נוחות לקריאה או ברשימות תבליטים.
3. מידע שלא מופיע: אם נשאלת שאלה רפואית שאין לגביה מידע בקטעים, ציין זאת בפשטות ובכנות, והמלץ להיוועץ ברופא המטפל.
4. שפה: ענה תמיד בעברית טבעית, רהוטה וזורמת.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])

GREETINGS = {"שלום", "היי", "הי", "בוקר טוב", "ערב טוב", "צהריים טובים", "מה קורה", "מה נשמע", "מי אתה", "תודה", "תודה רבה"}

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("שאל כל שאלה בנושא סוכרת או פתח בשיחה..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("מעבד תשובה..."):
            start_time = time.time()
            clean_q = user_query.strip().lower()

            # מענה מהיר ומותאם לשיחת פתיחה ללא הרצת RAG מיותרת
            if clean_q in GREETINGS or len(clean_q.split()) <= 2 and any(w in clean_q for w in ["שלום", "היי", "הי", "מה נשמע", "מה קורה"]):
                docs = []
                context = "אין צורך במקורות - שיחת פתיחה או ברכה."
                sources = []
            else:
                docs = retriever.invoke(user_query)
                # חיתוך מקטעים ארוכים למניעת עומס על ה-API
                context = "\n\n---\n\n".join([doc.page_content[:1500] for doc in docs])
                sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

            chain = prompt | llm
            response = chain.invoke({"context": context, "question": user_query})
            ans_text = response.content
            
            latency = time.time() - start_time
            
            st.markdown(ans_text)
            
            log_interaction(
                session_id=st.session_state.session_id,
                question=user_query,
                answer=ans_text,
                sources=sources,
                latency=latency
            )
            
    st.session_state.messages.append({"role": "assistant", "content": ans_text})
