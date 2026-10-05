import streamlit as st
import os
import sys
import time
import uuid
import gc
import torch
from datetime import datetime
from groq import Groq
from supabase import create_client, Client
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# אכיפת קידוד UTF-8 גלובלי למניעת שגיאות קידוד ASCII
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LANG"] = "C.UTF-8"
os.environ["LC_ALL"] = "C.UTF-8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# הגבלת משאבי CPU וזיכרון למניעת קריסות בענן
torch.set_num_threads(1)

# הגדרות Groq API
part1 = "gsk_gHj5VLlVTDHJbFJVbgFY"
part2 = "WGdyb3FYjPTo2EWiTiYgLqU9aGSrPT4l"
GROQ_API_KEY = part1 + part2
MODEL_NAME = "openai/gpt-oss-20b"
DB_DIR = "./chroma_db"

# הגדרות Supabase
SUPABASE_URL = "https://jscuqbkruilcpzxuedwm.supabase.co"
SUPABASE_KEY = "sb_publishable_omrw8g5n3OxadaDZZWPcjw_r_9aIWlJ"

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

# עיצוב CSS ותמיכת RTL מלאה
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

# אתחול משתני Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "demographics_completed" not in st.session_state:
    st.session_state.demographics_completed = False

if "messages" not in st.session_state:
    st.session_state.messages = []

# פונקציות שמירה ל-Supabase במבנה עמודות שטוח
def save_participant(participant_record: dict):
    try:
        supabase.table("participants").insert(participant_record).execute()
    except Exception as e:
        st.error(f"שגיאת שמירת נתוני שאלון: {str(e)}")

def log_interaction(session_id: str, question: str, answer: str, sources: list, latency: float):
    try:
        clean_question = question.encode('utf-8', 'ignore').decode('utf-8')
        clean_answer = answer.encode('utf-8', 'ignore').decode('utf-8')
        clean_sources = "; ".join(str(s) for s in sources) if sources else ""
        clean_sources = clean_sources.encode('utf-8', 'ignore').decode('utf-8')

        supabase.table("chat_interactions").insert({
            "session_id": str(session_id),
            "user_question": clean_question,
            "model_answer": clean_answer,
            "retrieved_sources": clean_sources,
            "latency_seconds": round(float(latency), 2)
        }).execute()
    except Exception as e:
        st.error(f"שגיאת תיעוד שיחה: {str(e)}")

# שלב 1: שאלון רקע מלא (19 שאלות)
if not st.session_state.demographics_completed:
    st.subheader("שאלון רקע קצר")
    st.write("אנא סמן/י את התשובה המתאימה ביותר בכל אחד מהסעיפים הבאים לפני תחילת השיחה:")

    with st.form("demographics_form"):
        # 1. גיל
        q1_age = st.radio(
            "1. גיל:",
            ["60–64", "65–69", "70–74", "75–79", "80 ומעלה"],
            index=None
        )

        # 2. מגדר
        q2_gender = st.radio(
            "2. מגדר:",
            ["גבר", "אישה", "אחר", "מעדיף/ה לא לציין"],
            index=None
        )

        # 3. השכלה
        q3_education = st.radio(
            "3. רמת השכלה:",
            [
                "ללא השכלה פורמלית",
                "השכלה תיכונית",
                "השכלה על־תיכונית / מקצועית",
                "תואר ראשון",
                "תואר שני ומעלה"
            ],
            index=None
        )

        # 4. מקום מגורים
        q4_residence = st.radio(
            "4. מקום מגורים:",
            ["עיר גדולה", "עיר קטנה / יישוב עירוני", "יישוב כפרי / מושב / קיבוץ"],
            index=None
        )

        # 5. תדירות שימוש באינטרנט
        q5_internet_freq = st.radio(
            "5. תדירות שימוש באינטרנט:",
            ["יומי", "מספר פעמים בשבוע", "לעיתים רחוקות", "כלל לא"],
            index=None
        )

        # 6. תדירות שימוש בסמארטפון
        q6_smartphone_freq = st.radio(
            "6. תדירות שימוש בסמארטפון:",
            ["יומי", "מספר פעמים בשבוע", "לעיתים רחוקות", "כלל לא"],
            index=None
        )

        # 7. ניסיון ב-AI
        q7_ai_experience = st.radio(
            "7. רמת ניסיון בשימוש במערכות בינה מלאכותית (AI):",
            [
                "אין ניסיון כלל",
                "ניסיון בסיסי (שימוש מועט או התנסות ראשונית)",
                "ניסיון בינוני (שימוש תקופתי)",
                "ניסיון מתקדם (שימוש קבוע ומגוון)"
            ],
            index=None
        )

        # 8. תדירות שימוש בצ'אטבוטים
        q8_chatbot_freq = st.radio(
            "8. תדירות שימוש בצ'אטבוטים מבוססי בינה מלאכותית (כגון ChatGPT, Gemini):",
            ["יומי", "מספר פעמים בשבוע", "לעיתים רחוקות", "כלל לא"],
            index=None
        )

        # 9. מצב בריאותי כללי
        q9_health_status = st.radio(
            "9. מצב בריאותי כללי (הערכתך האישית):",
            ["מצוין", "טוב", "בינוני", "ירוד"],
            index=None
        )

        # 10. חיפוש מידע רפואי באינטרנט
        q10_health_search_freq = st.radio(
            "10. באיזו תדירות את/ה מחפש/ת מידע רפואי באינטרנט?",
            ["לעיתים קרובות", "לעיתים", "לעיתים רחוקות", "אף פעם"],
            index=None
        )

        # 11. שפת אם
        q11_native_lang = st.radio(
            "11. שפת אם:",
            ["עברית", "ערבית", "רוסית", "אנגלית", "אחר"],
            index=None
        )
        q11_other = st.text_input("אם בחרת 'אחר' בשפת אם, פרט/י כאן:")

        # 12. שפת חיפוש באינטרנט
        q12_search_lang = st.radio(
            "12. באיזו שפה את/ה משתמש/ת לרוב בעת חיפוש מידע באינטרנט?",
            ["עברית", "אנגלית", "שפה אחרת"],
            index=None
        )
        q12_other = st.text_input("אם בחרת 'שפה אחרת' בחיפוש, פרט/י כאן:")

        # 13. שימוש במכשירים דיגיטליים
        q13_digital_devices = st.radio(
            "13. עד כמה את/ה משתמש/ת במכשירים דיגיטליים (מחשב, סמארטפון, טאבלט) ביום־יום?",
            ["כלל לא", "במידה מועטה", "במידה בינונית", "במידה רבה", "במידה רבה מאוד"],
            index=None
        )

        # 14. שימושים בטכנולוגיה (בחירה מרובה)
        st.markdown("**14. לאילו שימושים את/ה נעזר/ת בטכנולוגיה? (ניתן לסמן יותר מתשובה אחת):**")
        tech_comm = st.checkbox("תקשורת (WhatsApp, מיילים)")
        tech_search = st.checkbox("חיפוש מידע באינטרנט")
        tech_admin = st.checkbox("ניהול עניינים אישיים (בנק, קופות חולים וכו')")
        tech_content = st.checkbox("צריכת תוכן (חדשות, סרטונים)")
        tech_social = st.checkbox("רשתות חברתיות")
        tech_other = st.text_input("שימוש אחר:")

        # 15. ביטחון בטכנולוגיה חדשה
        q15_tech_confidence = st.radio(
            "15. עד כמה את/ה מרגיש/ה ביטחון ביכולת שלך להשתמש בטכנולוגיה חדשה (אפליקציות, אתרים חדשים)?",
            ["כלל לא בטוח/ה", "במידה מועטה", "במידה בינונית", "במידה רבה", "במידה רבה מאוד"],
            index=None
        )

        # 16. התמודדות עם קושי טכנולוגי
        q16_tech_difficulty = st.radio(
            "16. כאשר את/ה נתקל/ת בקושי טכנולוגי, כיצד את/ה נוהג/ת לפעול?",
            ["מוותר/ת", "מבקש/ת עזרה מאחרים", "מנסה לבד עד שמצליח/ה", "מחפש/ת פתרון באינטרנט", "אחר"],
            index=None
        )
        q16_other = st.text_input("אם בחרת 'אחר' בהתמודדות עם קושי, פרט/י כאן:")

        # 17. חשיבות הבנת אופן הפעולה
        q17_understand_tech = st.radio(
            "17. עד כמה חשוב לך להבין כיצד מערכות טכנולוגיות פועלות?",
            ["כלל לא חשוב", "במידה מועטה", "במידה בינונית", "במידה רבה", "במידה רבה מאוד"],
            index=None
        )

        # 18. אבחון סוכרת מסוג 2
        q18_diabetes_diagnosis = st.radio(
            "18. האם אובחנת בעבר עם סוכרת מסוג 2?",
            ["כן", "לא", "טרום־סוכרת"],
            index=None
        )

        # 19. סוכרת במשפחה
        q19_family_diabetes = st.radio(
            "19. האם יש לך בן/בת משפחה קרוב/ה המתמודד/ת עם סוכרת מסוג 2?",
            ["כן", "לא"],
            index=None
        )

        submitted = st.form_submit_button("סיום שאלון ומעבר לשיחה עם העוזר הקליני ←")

        if submitted:
            mandatory_checks = [
                q1_age, q2_gender, q3_education, q4_residence, q5_internet_freq,
                q6_smartphone_freq, q7_ai_experience, q8_chatbot_freq, q9_health_status,
                q10_health_search_freq, q11_native_lang, q12_search_lang, q13_digital_devices,
                q15_tech_confidence, q16_tech_difficulty, q17_understand_tech,
                q18_diabetes_diagnosis, q19_family_diabetes
            ]
            if any(item is None for item in mandatory_checks):
                st.warning("אנא השלם/י את כל השאלות לפני המעבר לשיחה.")
            else:
                tech_uses = []
                if tech_comm: tech_uses.append("תקשורת")
                if tech_search: tech_uses.append("חיפוש מידע")
                if tech_admin: tech_uses.append("ניהול עניינים אישיים")
                if tech_content: tech_uses.append("צריכת תוכן")
                if tech_social: tech_uses.append("רשתות חברתיות")
                if tech_other.strip(): tech_uses.append(f"אחר: {tech_other.strip()}")

                participant_record = {
                    "session_id": str(st.session_state.session_id),
                    "q1_age": q1_age,
                    "q2_gender": q2_gender,
                    "q3_education": q3_education,
                    "q4_residence": q4_residence,
                    "q5_internet_frequency": q5_internet_freq,
                    "q6_smartphone_frequency": q6_smartphone_freq,
                    "q7_ai_experience": q7_ai_experience,
                    "q8_chatbot_frequency": q8_chatbot_freq,
                    "q9_health_status": q9_health_status,
                    "q10_health_search_frequency": q10_health_search_freq,
                    "q11_native_language": q11_other.strip() if q11_native_lang == "אחר" and q11_other.strip() else q11_native_lang,
                    "q12_search_language": q12_other.strip() if q12_search_lang == "שפה אחרת" and q12_other.strip() else q12_search_lang,
                    "q13_digital_devices_frequency": q13_digital_devices,
                    "q14_technology_uses": ", ".join(tech_uses),
                    "q15_tech_confidence": q15_tech_confidence,
                    "q16_tech_difficulty_action": q16_other.strip() if q16_tech_difficulty == "אחר" and q16_other.strip() else q16_tech_difficulty,
                    "q17_importance_of_understanding_tech": q17_understand_tech,
                    "q18_diabetes_diagnosis": q18_diabetes_diagnosis,
                    "q19_family_diabetes": q19_family_diabetes
                }

                save_participant(participant_record)
                st.session_state.demographics_completed = True
                st.rerun()

# שלב 2: מסך הצ'אט
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
