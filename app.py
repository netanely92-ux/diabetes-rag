import streamlit as st
import os
import time
import uuid
import gc
import torch
from datetime import datetime
from groq import Groq
from supabase import create_client, Client
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# הגבלת משאבי CPU וזיכרון למניעת קריסות בענן
torch.set_num_threads(1)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
תאימות בסיסית

# הגדרות Groq API
part1 = "gsk_gHj5VLlVTDHJbFJVbgFY"
part2 = "WGdyb3FYjPTo2EWiTiYgLqU9aGSrPT4l"
GROQ_API_KEY = part1 + part2
MODEL_NAME = "openai/gpt-oss-20b"
DB_DIR = "./chroma_db"

# הגדרות Supabase
SUPABASE_URL = "https://jscuqbkruilcpzxuedwm.supabase.co"
# הדבק כאן את ה-Publishable Key שהעתקת:
SUPABASE_KEY = "הדבק_כאן_את_ה_KEY_שלך"

@st.cache_resource
def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = get_supabase_client()

st.set_page_config(
    page_title="עוזר סוכרת קליני",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# הגדרת כיווניות RTL ועיצוב קליני
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@300;400;500;600;700&display=swap');

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

    .hero-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.75rem;
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
    }
    .hero-card p {
        margin: 0.4rem 0 0 0;
        font-size: 0.95rem;
        color: #64748b;
    }

    /* טופס שאלון */
    .stForm {
        background: #ffffff !important;
        padding: 2rem !important;
        border-radius: 16px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03) !important;
    }

    div[data-testid="stChatMessage"] {
        direction: rtl !important;
        text-align: right !important;
        padding: 1.1rem 1.35rem !important;
        border-radius: 16px !important;
        margin-bottom: 1rem !important;
        font-size: 1.05rem !important;
        line-height: 1.7 !important;
    }

    div[data-testid="stChatMessageContent"] * {
        direction: rtl !important;
        text-align: right !important;
        unicode-bidi: plaintext !important;
    }

    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background-color: #eff6ff !important;
        border: 1px solid #dbeafe !important;
        border-right: 4px solid #3b82f6 !important;
    }

    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-right: 4px solid #10b981 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03) !important;
    }

    div[data-testid="stChatInput"] textarea {
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Rubik', sans-serif !important;
        font-size: 1rem !important;
        unicode-bidi: plaintext !important;
    }

    #MainMenu, footer, header {
        visibility: hidden !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-card">
    <h2>🩺 עוזר מידע קליני בנושא סוכרת</h2>
    <p>מערכת מענה מבוססת מידע רפואי בנושאי מניעה, תסמינים, תזונה וטיפול</p>
</div>
""", unsafe_allow_html=True)

# אתחול Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "demographics_completed" not in st.session_state:
    st.session_state.demographics_completed = False

if "messages" not in st.session_state:
    st.session_state.messages = []

# פונקציות שמירה ל-Supabase
def save_participant(session_id: str, demographics_data: dict):
    try:
        supabase.table("participants").insert({
            "session_id": session_id,
            "demographics": demographics_data
        }).execute()
    except Exception as e:
        st.error(f"שגיאת שמירת משתתף: {e}")

def log_interaction(session_id: str, question: str, answer: str, sources: list, latency: float):
    try:
        supabase.table("chat_interactions").insert({
            "session_id": session_id,
            "user_question": question,
            "model_answer": answer,
            "retrieved_sources": "; ".join(sources),
            "latency_seconds": round(latency, 2)
        }).execute()
    except Exception as e:
        st.error(f"שגיאת תיעוד שיחה: {e}")

# שלב 1: שאלון מקדים
if not st.session_state.demographics_completed:
    st.subheader("שאלון מקדים קצר")
    st.write("לפני תחילת השיחה, אנא ענה על מספר שאלות רקע קצרות לצורכי המחקר:")

    with st.form("demographics_form"):
        age_group = st.selectbox(
            "קבוצת גיל:",
            ["בחר/י...", "מתחת ל-25", "25-34", "35-44", "45-54", "55-64", "65 ומעלה"]
        )
        gender = st.selectbox(
            "מגדר:",
            ["בחר/י...", "אישה", "גבר", "אחר / מעדיף לא לציין"]
        )
        diabetes_relation = st.selectbox(
            "מהי מידת ההיכרות שלך עם תחום הסוכרת?",
            [
                "בחר/י...",
                "אין לי היכרות אישית",
                "בן משפחה או חבר קרוב מתמודד עם סוכרת",
                "אני מאובחן/ת עם טרום סוכרת או סוכרת",
                "רקע מקצועי / עולם הבריאות"
            ]
        )
        ai_experience = st.select_slider(
            "עד כמה את/ה משתמש/ת בכלי בינה מלאכותית (כגון ChatGPT) בחיי היומיום?",
            options=["כלל לא", "לעיתים רחוקות", "מדי פעם", "לעיתים קרובות", "באופן קבוע ויומיומי"]
        )

        submitted = st.form_submit_button("המשך לשיחה עם העוזר הקליני ←")

        if submitted:
            if age_group == "בחר/י..." or gender == "בחר/י..." or diabetes_relation == "בחר/י...":
                st.warning("אנא השלם/י את כל השדות לפני המעבר לשיחה.")
            else:
                demo_data = {
                    "age_group": age_group,
                    "gender": gender,
                    "diabetes_relation": diabetes_relation,
                    "ai_experience": ai_experience
                }
                save_participant(st.session_state.session_id, demo_data)
                st.session_state.demographics_completed = True
                st.rerun()

# שלב 2: מסך הצ'אט (מוצג רק לאחר מילוי השאלון)
else:
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
2. שאלות קליניות: ספק תשובה מנומקת, ברורה ומאורגנת היטב על בסיס המידע הרפואי הנתון.
3. סייג רפואי: אם המידע אינו מופיע במאגר, ציין זאת ישירות והמלץ להיוועץ ברופא המטפל או בצוות הרפואי.
4. שפה וסגנון: כתוב בעברית תקנית, עשירה ומקצועית בלבד.
"""

    GREETINGS = {"שלום", "היי", "הי", "בוקר טוב", "ערב טוב", "צהריים טובים", "מה קורה", "מה נשמע", "מי אתה", "תודה", "תודה רבה"}

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
                st.markdown(f"<div dir='rtl' style='text-align: right;'>{ans_text}</div>", unsafe_allow_html=True)

                log_interaction(
                    session_id=st.session_state.session_id,
                    question=user_query,
                    answer=ans_text,
                    sources=sources,
                    latency=latency
                )

        st.session_state.messages.append({"role": "assistant", "content": ans_text})
