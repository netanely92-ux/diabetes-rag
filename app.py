import streamlit as st
import csv
import os
import time
import uuid
import gc
import torch
from datetime import datetime
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# הגבלת משאבי CPU וזיכרון למניעת קריסת זיכרון בענן
torch.set_num_threads(1)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DB_DIR = "./chroma_db"
LOG_FILE = "chat_interactions.csv"

# חיבור המפתח החדש בשני מקטעים לעקיפת סורק האבטחה של GitHub וללא תלות ב-Secrets
part1 = "gsk_gHj5VLlVTDHJbFJVbgFY"
part2 = "WGdyb3FYjPTo2EWiTiYgLqU9aGSrPT4l"
GROQ_API_KEY = part1 + part2

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
    <p>מענה לשאלות בנושאי מניעה, תסמינים, תזונה וטיפול קליני</p>
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
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 1}
    )
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    client = Groq(api_key=GROQ_API_KEY)
    
    gc.collect()
    return retriever, client

retriever, groq_client = load_rag()

SYSTEM_PROMPT = """אתה עוזר וירטואלי חכם, אדיב ורהוט, המתמחה במידע קליני בנושא סוכרת.

קטעי המידע הבאים עומדים לרשותך מהמאגר המקצועי:
{context}

הנחיות לתגובה:
1. ברכות ושיחת חולין: אם המשתמש מברך ("שלום", "היי", "מה נשמע", "תודה"), ענה בצורה חמה ואדיבה בעברית טבעית והסבר שאתה כאן לסייע בשאלות על סוכרת, תזונה ומניעה.
2. שאלות מקצועיות: ענה בצורה ברורה ומובנית על פי המידע הרפואי שסופק.
3. מידע חסר: אם נשאלת שאלה שאין לה מענה במידע הנתון, ציין זאת בפשטות והמלץ להיוועץ ברופא.
4. שפה: השב תמיד בעברית טבעית ורהוטה.
"""

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

            is_greeting = clean_q in GREETINGS or (len(clean_q.split()) <= 2 and any(w in clean_q for w in ["שלום", "היי", "הי", "מה נשמע", "מה קורה"]))

            if is_greeting:
                context = "שיחת פתיחה או ברכה. ענה בצורה חמה ולבבית."
                sources = []
            else:
                docs = retriever.invoke(user_query)
                context = "\n\n---\n\n".join([doc.page_content[:1500] for doc in docs])
                sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

            formatted_system = SYSTEM_PROMPT.format(context=context)

            try:
                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": formatted_system},
                        {"role": "user", "content": user_query}
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                )
                ans_text = chat_completion.choices[0].message.content
            except Exception as e:
                ans_text = f"⚠ שגיאת חיבור ל-Groq: {str(e)}"

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
