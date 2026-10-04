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

# הרכבת המפתח
part1 = "gsk_gHj5VLlVTDHJbFJVbgFY"
part2 = "WGdyb3FYjPTo2EWiTiYgLqU9aGSrPT4l"
GROQ_API_KEY = part1 + part2

# מודל שיחה פעיל מתוך הרשימה בחשבונך
MODEL_NAME = "openai/gpt-oss-20b"

st.set_page_config(
    page_title="עוזר סוכרת קליני",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# שדרוג עיצוב מלא - מודרני, נקי ו-RTL מוקפד
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Assistant:wght@300;400;500;600;700&display=swap');

    /* בסיס האתר וטיפוגרפיה */
    html, body, [class*="css"], .stApp {
        font-family: 'Assistant', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #f8fafc !important;
        direction: rtl !important;
        text-align: right !important;
        color: #1e293b !important;
    }

    /* כותרת עליונה בסגנון Card קליני יוקרתי */
    .hero-card {
        background: #ffffff;
        border-radius: 18px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        border-top: 4px solid #0284c7;
        text-align: right;
    }
    .hero-card h2 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .hero-card p {
        margin: 0.4rem 0 0 0;
        font-size: 0.95rem;
        color: #64748b;
        font-weight: 400;
        line-height: 1.5;
    }

    /* עיצוב בועות ההודעות בצ'אט */
    div[data-testid="stChatMessage"] {
        direction: rtl !important;
        text-align: right !important;
        padding: 1.1rem 1.35rem !important;
        border-radius: 16px !important;
        margin-bottom: 1rem !important;
        font-size: 1.02rem !important;
        line-height: 1.7 !important;
        max-width: 90% !important;
    }

    /* הודעת משתמש */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background-color: #f0fdfa !important;
        border: 1px solid #ccfbf1 !important;
        border-right: 4px solid #0d9488 !important;
        margin-left: auto !important;
        margin-right: 0 !important;
    }

    /* הודעת עוזר קליני */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-right: 4px solid #0284c7 !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03) !important;
        margin-right: auto !important;
    }

    /* שדרוג שדה ההקלדה (Chat Input) */
    div[data-testid="stChatInput"] {
        direction: rtl !important;
        padding-bottom: 1.5rem !important;
    }
    div[data-testid="stChatInput"] > div {
        border-radius: 16px !important;
        border: 1px solid #cbd5e1 !important;
        background-color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stChatInput"] > div:focus-within {
        border-color: #0284c7 !important;
        box-shadow: 0 4px 18px rgba(2, 132, 199, 0.15) !important;
    }
    div[data-testid="stChatInput"] textarea {
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Assistant', sans-serif !important;
        font-size: 1rem !important;
        color: #0f172a !important;
        padding: 0.8rem 1rem !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #94a3b8 !important;
        font-weight: 400 !important;
    }

    /* רשימות ותבליטים */
    ul, ol {
        direction: rtl !important;
        text-align: right !important;
        padding-right: 1.5rem !important;
        padding-left: 0 !important;
        margin: 0.6rem 0 !important;
    }
    li {
        margin-bottom: 0.35rem !important;
    }

    /* הסתרת רכיבי מערכת מיותרים */
    #MainMenu, footer, header {
        visibility: hidden !important;
    }
</style>
""", unsafe_allow_html=True)

# כותרת ראשית מעוצבת
st.markdown("""
<div class="hero-card">
    <h2>🩺 עוזר מידע קליני לסוכרת</h2>
    <p>מערכת מבוססת ידע קליני למענה מקצועי בנושאי מניעה, תסמינים, תזונה ואיזון רפואי</p>
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

SYSTEM_PROMPT = """אתה עוזר וירטואלי מומחה, מקצועי, אדיב ורהוט, המתמחה במידע קליני בנושא סוכרת.

קטעי המידע הבאים עומדים לרשותך מהמאגר המקצועי:
{context}

הנחיות קפדניות לתגובה:
1. ברכות ושיחת חולין: אם המשתמש מברך ("שלום", "היי", "מה נשמע", "תודה"), ענה בצורה מכבדת, חמה ולבבית בעברית טבעית, והסבר שאתה כאן לסייע בכל שאלה בנושאי סוכרת, איזון סוכר ותזונה.
2. שאלות קליניות: ספק תשובה מנומקת, ברורה ומאורגנת היטב (השתמש בנקודות או פסקאות קצרות במידת הצורך) על בסיס המידע הרפואי הנתון.
3. סייג רפואי: אם המידע אינו מופיע במאגר, ציין זאת ישירות והמלץ להיוועץ ברופא המטפל או בצוות הרפואי.
4. שפה וסגנון: כתוב בעברית תקנית, עשירה ומקצועית.
"""

GREETINGS = {"שלום", "היי", "הי", "בוקר טוב", "ערב טוב", "צהריים טובים", "מה קורה", "מה נשמע", "מי אתה", "תודה", "תודה רבה"}

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("שאל שאלה בנושא סוכרת, מדדים, תזונה או פתח בשיחה..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("מעבד מידע קליני..."):
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
                    model=MODEL_NAME,
                    temperature=0.3,
                )
                ans_text = chat_completion.choices[0].message.content
            except Exception as e:
                ans_text = f"⚠ שגיאה: {str(e)}"

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
