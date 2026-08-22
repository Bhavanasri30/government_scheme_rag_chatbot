import logging
import re
from uuid import uuid4

import streamlit as st

from rag import ask_schemesathi

# ---------------------------------------------------------------------------
# Page config MUST be the first Streamlit command in the script.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="YojanaSetu | Scheme Eligibility Assistant",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

ASSISTANT_AVATAR = "🤖"
USER_AVATAR = "👤"

SUGGESTIONS = (
    ("🎓", "Scholarships for college students"),
    ("🌾", "Government schemes for farmers"),
    ("👩‍💼", "Support for women entrepreneurs"),
    ("🏠", "Benefits for low-income families"),
)

PROFILE_DEFAULTS = {
    "age": "",
    "gender": "Not provided",
    "state": "",
    "category": "Not provided",
    "occupation": "",
    "education": "",
    "annual_income": "",
}


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------
def create_chat():
    chat_id = uuid4().hex
    st.session_state.chats[chat_id] = {"title": "New conversation", "messages": []}
    st.session_state.active_chat_id = chat_id
    return chat_id


def active_chat():
    return st.session_state.chats[st.session_state.active_chat_id]


def delete_chat(chat_id):
    st.session_state.chats.pop(chat_id, None)
    if not st.session_state.chats:
        create_chat()
    elif st.session_state.active_chat_id == chat_id:
        st.session_state.active_chat_id = next(iter(st.session_state.chats))


def title_from_question(question):
    words = re.findall(r"\S+", question.strip())
    title = " ".join(words[:7])
    if len(words) > 7 or len(title) > 35:
        title = title[:35].rstrip() + "..."
    return title or "New conversation"


def build_complete_question(question):
    profile = st.session_state.profile_data
    return f"""User profile:
Age: {profile.get('age') or 'Not provided'}
Gender: {profile.get('gender') or 'Not provided'}
State: {profile.get('state') or 'Not provided'}
Category: {profile.get('category') or 'Not provided'}
Occupation: {profile.get('occupation') or 'Not provided'}
Education: {profile.get('education') or 'Not provided'}
Annual family income: {profile.get('annual_income') or 'Not provided'}

User question:
{question}"""


def submit_question(question):
    """Appends the user + assistant messages to the active chat. Does not
    render anything itself -- the caller triggers a single st.rerun()
    afterward so the conversation is drawn exactly once."""
    chat = active_chat()
    chat["messages"].append({"role": "user", "content": question})
    if len([m for m in chat["messages"] if m["role"] == "user"]) == 1:
        chat["title"] = title_from_question(question)

    with st.spinner("Searching the government scheme knowledge base..."):
        try:
            answer = ask_schemesathi(build_complete_question(question))
        except Exception:
            LOGGER.exception("SchemeSathi backend request failed")
            answer = (
                "I could not search the scheme knowledge base right now. "
                "Please wait a moment and try again."
            )

    chat["messages"].append({"role": "assistant", "content": answer})


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_profile():
    with st.expander("👤 Your profile", expanded=False):
        st.caption(
            "Profile information improves preliminary eligibility matching and "
            "is used only during the current session."
        )
        profile = st.session_state.profile_data
        profile["age"] = st.text_input("Age", key="profile_age", placeholder="e.g. 25")
        profile["gender"] = st.selectbox(
            "Gender", ["Not provided", "Female", "Male", "Other"], key="profile_gender"
        )
        profile["state"] = st.text_input(
            "State or Union Territory", key="profile_state", placeholder="e.g. Maharashtra"
        )
        profile["category"] = st.selectbox(
            "Social category",
            ["Not provided", "General", "OBC", "SC", "ST", "Minority"],
            key="profile_category",
        )
        profile["occupation"] = st.text_input(
            "Occupation", key="profile_occupation", placeholder="e.g. Student"
        )
        profile["education"] = st.text_input(
            "Education level", key="profile_education", placeholder="e.g. B.Tech"
        )
        profile["annual_income"] = st.text_input(
            "Annual family income", key="profile_income", placeholder="e.g. 200000"
        )


def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="brand-mark">YS</div>'
            '<div><div class="brand-name">YojanaSetu</div>'
            '<div class="brand-subtitle">Government Schemes Assistant</div></div></div>',
            unsafe_allow_html=True,
        )

        if st.button("＋ New Chat", use_container_width=True, type="primary"):
            create_chat()
            st.rerun()

        st.markdown('<div class="sidebar-heading">Recent Chats</div>', unsafe_allow_html=True)
        for chat_id, chat in list(st.session_state.chats.items()):
            selected = chat_id == st.session_state.active_chat_id
            select_col, delete_col = st.columns([5, 1])
            with select_col:
                if st.button(
                    chat["title"],
                    key=f"select_{chat_id}",
                    use_container_width=True,
                    type="primary" if selected else "secondary",
                ):
                    if not selected:
                        st.session_state.active_chat_id = chat_id
                        st.rerun()
            with delete_col:
                if st.button("✕", key=f"delete_{chat_id}", help="Delete this chat"):
                    delete_chat(chat_id)
                    st.rerun()

        if len(st.session_state.chats) > 1:
            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
            if not st.session_state.get("confirm_clear"):
                if st.button("Clear all chats", use_container_width=True):
                    st.session_state.confirm_clear = True
                    st.rerun()
            else:
                st.warning("Delete every saved conversation? This cannot be undone.")
                confirm_col, cancel_col = st.columns(2)
                with confirm_col:
                    if st.button(
                        "Yes, clear",
                        key="confirm_clear_yes",
                        use_container_width=True,
                        type="primary",
                    ):
                        st.session_state.chats = {}
                        create_chat()
                        st.session_state.confirm_clear = False
                        st.rerun()
                with cancel_col:
                    if st.button("Cancel", key="confirm_clear_cancel", use_container_width=True):
                        st.session_state.confirm_clear = False
                        st.rerun()

        render_profile()


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "profile_data" not in st.session_state:
    st.session_state.profile_data = PROFILE_DEFAULTS.copy()
for key, value in PROFILE_DEFAULTS.items():
    st.session_state.profile_data.setdefault(key, value)
for profile_key, widget_key in {
    "age": "profile_age",
    "gender": "profile_gender",
    "state": "profile_state",
    "category": "profile_category",
    "occupation": "profile_occupation",
    "education": "profile_education",
    "annual_income": "profile_income",
}.items():
    st.session_state.setdefault(widget_key, st.session_state.profile_data[profile_key])
if "active_chat_id" not in st.session_state or st.session_state.active_chat_id not in st.session_state.chats:
    create_chat()
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --ys-bg: #080B14;
        --ys-sidebar: #0D1220;
        --ys-surface: #121827;
        --ys-surface-soft: #171E30;
        --ys-border: #27324A;
        --ys-primary: #7C6FF6;
        --ys-primary-hover: #9186FF;
        --ys-accent: #4FD1C5;
        --ys-text: #F5F7FC;
        --ys-muted: #9BA8BE;
    }
    .stApp { background: var(--ys-bg); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: var(--ys-sidebar); border-right: 1px solid var(--ys-border); }
    [data-testid="stSidebar"] > div:first-child { padding: 1.4rem 1rem; }
    .main .block-container { max-width: 920px; padding: 3rem 1.5rem 7rem; }
    .brand { display: flex; gap: .7rem; align-items: center; margin-bottom: 1.8rem; }
    .brand-mark { display: grid; place-items: center; width: 2.25rem; height: 2.25rem; border-radius: 9px; background: linear-gradient(135deg, var(--ys-primary), #5A4FD8); color: white; font-size: .78rem; font-weight: 800; box-shadow: 0 8px 24px rgba(124,111,246,.25); }
    .brand-name { color: var(--ys-text); font-size: 1.15rem; font-weight: 700; }
    .brand-subtitle, .sidebar-heading { color: var(--ys-muted); font-size: .78rem; }
    .sidebar-heading { margin: 1.8rem 0 .6rem; text-transform: uppercase; letter-spacing: .08em; }
    [data-testid="stSidebar"] .stButton button { text-align: left; border-radius: 9px; border: 1px solid transparent; color: #DCE2EE; background: transparent; box-shadow: none; }
    [data-testid="stSidebar"] .stButton button:hover { border-color: var(--ys-primary); background: rgba(124,111,246,.10); color: white; }
    [data-testid="stSidebar"] .stButton button[kind="primary"] { text-align: center; background: var(--ys-primary); color: white; border: 0; }
    [data-testid="stSidebar"] .stButton button[kind="primary"]:hover { background: var(--ys-primary-hover); }
    [data-testid="stSidebar"] [data-testid="stExpander"] { margin-top: 1.2rem; border-color: var(--ys-border); background: var(--ys-surface); }
    [data-testid="stChatMessage"] { border: 1px solid var(--ys-border); border-radius: 14px; padding: 1rem 1.2rem; margin: 1rem 0; background: var(--ys-surface); color: #E8ECF4; box-shadow: 0 10px 30px rgba(0,0,0,.18); }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { background: #171D35; border-color: #34345E; }
    [data-testid="stChatInput"] { background: var(--ys-surface-soft); border: 1px solid #35415C; border-radius: 14px; box-shadow: 0 14px 40px rgba(0,0,0,.28); }
    [data-testid="stChatInput"] textarea { color: var(--ys-text); }
    [data-testid="stChatInput"] textarea::placeholder { color: var(--ys-muted); opacity: 1; }
    [data-testid="stChatInput"]:focus-within { border-color: var(--ys-primary); box-shadow: 0 0 0 1px var(--ys-primary), 0 14px 40px rgba(0,0,0,.28); }
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: #DEE4EF; line-height: 1.65; }
    h1, h2, h3 { color: var(--ys-text); }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] .stCaption { color: #B0BBCD !important; }
    [data-baseweb="input"], [data-baseweb="select"] > div { background: #0A0F1B; border-color: #35415C; color: #E8ECF4; }
    .main .stButton button { border-radius: 11px; border: 1px solid var(--ys-border); background: var(--ys-surface); color: #E6EAF2; }
    .main .stButton button:hover { border-color: var(--ys-primary); background: rgba(124,111,246,.10); color: white; }
    footer, #MainMenu { visibility: hidden; }
    .welcome { padding: 4rem 0 1.5rem; text-align: center; }
    .welcome-kicker { color: var(--ys-accent); font-weight: 700; letter-spacing: .12em; text-transform: uppercase; font-size: .75rem; }
    .welcome h1 { font-size: clamp(1.8rem, 4vw, 2.6rem); margin: .7rem 0 .6rem; color: var(--ys-text); }
    .welcome p { color: var(--ys-muted); margin: 0 auto; max-width: 560px; }
    .status-badge { display: inline-flex; align-items: center; gap: .4rem; margin-top: 1.1rem; padding: .3rem .8rem; border: 1px solid rgba(79,209,197,.22); border-radius: 999px; background: rgba(79,209,197,.09); color: var(--ys-accent); font-size: .75rem; font-weight: 600; }
    .status-badge::before { content: ""; width: .45rem; height: .45rem; border-radius: 50%; background: var(--ys-accent); box-shadow: 0 0 12px rgba(79,209,197,.55); }
    .suggestion-label { color: var(--ys-muted); font-size: .8rem; margin: 2.2rem 0 .7rem; text-align: center; }
    .disclaimer { border-top: 1px solid var(--ys-border); color: #8491A7; font-size: .76rem; line-height: 1.55; margin-top: 3rem; padding-top: 1rem; text-align: center; }
    @media (max-width: 640px) { .main .block-container { padding: 1.5rem 1rem 6rem; } .welcome { padding: 3rem 0 1rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
render_sidebar()
chat = active_chat()
user_messages = [m for m in chat["messages"] if m["role"] == "user"]

if not user_messages:
    st.markdown(
        '<div class="welcome"><div class="welcome-kicker">YojanaSetu</div>'
        "<h1>Find the right government scheme</h1>"
        "<p>Ask naturally. YojanaSetu retrieves relevant schemes and explains "
        "eligibility, benefits, documents and application steps.</p>"
        '<div class="status-badge">RAG knowledge base ready</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="suggestion-label">Try asking about</div>', unsafe_allow_html=True)
    suggestion_columns = st.columns(2)
    for index, (emoji, label) in enumerate(SUGGESTIONS):
        with suggestion_columns[index % 2]:
            if st.button(
                f"{emoji} {label}",
                key=f"suggestion_{index}",
                use_container_width=True,
            ):
                st.session_state.pending_question = label

for message in chat["messages"]:
    avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

typed_question = st.chat_input("Ask about government schemes...", key="chat_input")

# A single, unambiguous path for turning "the user just asked something" into
# "messages appended, then one rerun" -- whether the question came from a
# suggestion button or the chat input box. This guarantees nothing is ever
# submitted or rendered twice.
question_to_submit = None
if typed_question and typed_question.strip():
    question_to_submit = typed_question.strip()
elif st.session_state.pending_question:
    question_to_submit = st.session_state.pending_question

if question_to_submit:
    st.session_state.pending_question = None
    submit_question(question_to_submit)
    st.rerun()

st.markdown(
    '<div class="disclaimer">YojanaSetu provides preliminary guidance based on the available '
    "scheme dataset. Verify current eligibility, deadlines and application details through the "
    "official government portal.</div>",
    unsafe_allow_html=True,
)
