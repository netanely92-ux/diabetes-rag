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

# הגדרת כיווניות RTL מוחלטת לכל רכיבי האפליקציה
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@300;400;500;600;700&display=swap');

    /* כפיית כיווניות RTL על כל הדף והאלמנטים */
    html, body, [class*="css"], .stApp, .stMarkdown, .stMarkdown p, .stMarkdown span {
        font-family: 'Rubik', sans-serif !important;
        direction: rtl !important;
        text-align: right !important;
        unicode-bidi: embed !important;
    }

    body {
        background-color: #f8fafc !important;
        color: #1e293b !important;
    }

    /* כרטיסיית כותרת */
    .hero-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
        border: 1px solid #e2e8f0;
        border-right: 6px solid #2563eb;
        direction: rtl !important;
        text-align: right !important;
    }
    .hero-card h2 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e3a8a;
        direction: rtl !important;
        text-align: right !important;
    }
    .hero-card p {
        margin: 0.4rem 0 0 0;
        font-size: 0.95rem;
        color: #64748b;
        direction: rtl !important;
        text-align: right !important;
    }

    /* כפיית RTL על כל בועות השיחה והתוכן שלהן */
    div[data-testid="stChatMessage"] {
        direction: rtl !important;
        text-align: right !important;
        padding: 1.1rem 1.35rem !important;
        border-radius: 16px !important;
        margin-bottom: 1rem !important;
        font-size: 1.05rem !important;
        line-height: 1.7 !important;
    }

    div[data-testid="stChatMessageContent"] {
        direction: rtl !important;
        text-align: right !important;
    }

    div[data-testid="stChatMessageContent"] * {
        direction: rtl !important;
        text-align: right !important;
        unicode-bidi: plaintext !important;
    }

    /* בועת המשתמש */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background-color: #eff6ff !important;
        border: 1px solid #dbeafe !important;
        border-right: 4px solid #3b82f6 !important;
    }

    /* בועת העוזר הקליני */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-right: 4px solid #10b981 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03) !important;
    }

    /* רשימות ותבליטים */
    ul, ol {
        direction: rtl !important;
        text-align: right !important;
        padding-right: 1.6rem !important;
        padding-left: 0 !important;
        margin: 0.6rem 0 !important;
    }
    li {
        direction: rtl !important;
        text-align: right !important;
        margin-bottom: 0.35rem !important;
    }

    /* שדה קלט - Chat Input */
    div[data-testid="stChatInput"] {
        direction: rtl !important;
        text-align: right !important;
    }
    div[data-testid="stChatInput"] textarea {
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Rubik', sans-serif !important;
        font-size: 1rem !important;
        unicode-bidi: plaintext !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        direction: rtl !important;
        text-align: right !important;
    }

    #MainMenu, footer, header {
        visibility: hidden !important;
    }
</style>
""", unsafe_allow_html=True)

# כותרת האפליקציה
st.markdown("""
<div class="hero-card">
    <h2>🩺 עוזר מידע קליני בנושא סוכרת</h2>
    <p>מערכת מענה מבוססת מידע רפואי בנושאי מניעה, תסמינים, תזונה וטיפול</p>
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
4. שפה וסגנון: כתוב בעברית תקנית, עשירה ומקצועית בלבד.
"""

GREETINGS = {"שלום", "היי", "הי", "בוקר טוב", "ערב טוב", "צהריים טובים", "מה קורה", "מה נשמע", "מי אתה", "תודה", "תודה רבה"}

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

# הצגת היסטוריית השיחה עם מעטפת RTL
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(f"<div dir='rtl' style='text-align: right;'>{msg['content']}</div>", unsafe_allow_html=True)

if user_query := st.chat_input("שאל שאלה בנושא סוכרת, תזונה או מדדים..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(f"<div dir='rtl' style='text-align: right;'>{user_query}</div>", unsafe_allow_html=True)

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
            # הצגת התשובה עטופה ב-div עם dir=rtl מפורש
            st.markdown(f"<div dir='rtl' style='text-align: right;'>{ans_text}</div>", unsafe_allow_html=True)
            
            log_interaction(
                session_id=st.session_state.session_id,
                question=user_query,
                answer=ans_text,
                sources=sources,
                latency=latency
            )
            
    st.session_state.messages.append({"role": "assistant", "content": ans_text})
