import streamlit as st
import os
import textwrap
import fitz
import smtplib
import tempfile
from email.message import EmailMessage
from fpdf import FPDF
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

api_key = st.secrets["GEMINI"]["API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

st.set_page_config(page_title="Smart Document Summarizer and Q&A Bot", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Orbitron', sans-serif;
        background-color: #0f0f0f;
        color: #d0d0d0;
    }
    h1, h2, h3, h4 {
        text-align: center;
        color: #10f0ff;
        text-shadow: 0 0 5px #0ff;
    }
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(90deg, #0ff 0%, #09f 100%);
        color: #000;
        border: none;
        padding: 0.5em 1.2em;
        border-radius: 10px;
        font-weight: bold;
        box-shadow: 0 0 10px #0ff, 0 0 20px #09f;
        transition: transform 0.2s ease-in-out;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px #0ff, 0 0 25px #0ff;
    }
    .stTextInput>div>div>input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid #0ff;
        border-radius: 8px;
        color: #fff;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🤖 Smart Document Summarizer and Q&A Bot</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #ccc;'>Made by Vanshika Dureja | vanshika123d@gmail.com</h5>", unsafe_allow_html=True)
st.markdown('<hr style="border: 1px solid #0ff;">', unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "chunks_vectors" not in st.session_state:
    st.session_state.chunks_vectors = None
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "quiz" not in st.session_state:
    st.session_state.quiz = ""
if "auto_email" not in st.session_state:
    st.session_state.auto_email = False

def chunk_text(text, chunk_size=300):
    return textwrap.wrap(text, width=chunk_size)

def process_file(uploaded_file, file_type):
    if file_type == "pdf":
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
    else:
        text = uploaded_file.read().decode("utf-8")
    chunks = chunk_text(text)
    st.session_state.chunks = chunks
    vectorizer = TfidfVectorizer()
    chunks_vectors = vectorizer.fit_transform(chunks)
    st.session_state.vectorizer = vectorizer
    st.session_state.chunks_vectors = chunks_vectors

def get_relevant_chunks(query):
    vec = st.session_state.vectorizer.transform([query])
    similarities = cosine_similarity(vec, st.session_state.chunks_vectors).flatten()
    top_indices = similarities.argsort()[-3:][::-1]
    return "\n\n".join([st.session_state.chunks[i] for i in top_indices])

def build_prompt(query):
    chat_history = "\n".join([f"User: {q}\nBot: {a}" for q, a in st.session_state.history[-3:]])
    context = get_relevant_chunks(query)
    prompt = f"""You are a helpful assistant that answers questions based on the provided context.
Context: {context}
{chat_history}
User: {query}
Bot:"""
    return prompt

def summarize_document():
    joined_text = "\n".join(st.session_state.chunks[:10])
    prompt = f"""Summarize the following document clearly and concisely:
{joined_text}"""
    response = model.generate_content(prompt)
    st.session_state.summary = response.text.strip()

def generate_quiz():
    joined_text = "\n".join(st.session_state.chunks[:10])
    prompt = f"""
Based on the following content, generate a short multiple choice quiz with 3 questions to test understanding.
Each question should have 4 options and mark the correct answer clearly:
{joined_text}"""
    response = model.generate_content(prompt)
    st.session_state.quiz = response.text.strip()

def generate_pdf(summary, quiz):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"Summary:\n{summary}\n\nQuiz:\n{quiz}")
    path = tempfile.mktemp(suffix=".pdf")
    pdf.output(path)
    return path

def send_email(receiver_email, subject, content, attachment_path=None):
    sender_email = st.secrets["EMAIL"]["SENDER"]
    sender_password = st.secrets["EMAIL"]["PASSWORD"]
    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.set_content(content)
    if attachment_path:
        with open(attachment_path, "rb") as f:
            file_data = f.read()
            msg.add_attachment(file_data, maintype="application", subtype="pdf", filename="report.pdf")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, sender_password)
        server.send_message(msg)

def export_chat_history():
    log = "\n\n".join([f"User: {q}\nBot: {a}" for q, a in st.session_state.history])
    return log.encode("utf-8")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload & Process File")
    uploaded_file = st.file_uploader("Upload a file (.pdf or .txt)", type=["pdf", "txt"])
    if uploaded_file:
        file_type = uploaded_file.type.split("/")[-1]
        process_file(uploaded_file, file_type)
        st.success("✅ File processed successfully!")

with col2:
    st.subheader("💬 Ask a Question")
    user_query = st.text_input("Type your question:")
    if st.button("Ask"):
        if user_query.strip() and st.session_state.vectorizer and st.session_state.chunks:
            prompt = build_prompt(user_query)
            st.code(prompt)
            response = model.generate_content(prompt)
            bot_reply = response.text.strip()
            st.session_state.history.append((user_query, bot_reply))
            st.success("✅ Response Generated!")
            st.markdown(f"**🤖 Bot:** {bot_reply}")

st.subheader("📚 Actions")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📄 Summarize Document") and st.session_state.chunks:
        summarize_document()
        with st.expander("📑 View Summary"):
            st.write(st.session_state.summary)

with col2:
    if st.button("📝 Generate Quiz") and st.session_state.chunks:
        generate_quiz()
        with st.expander("🧪 View Quiz"):
            st.markdown(st.session_state.quiz)
        if st.session_state.auto_email:
            path = generate_pdf(st.session_state.summary, st.session_state.quiz)
            send_email(st.session_state.email_to_send, "Auto-Report: Summary & Quiz", "Auto-sent by Smart Bot", path)
            st.success("📬 Report auto-emailed!")

with col3:
    if st.session_state.summary and st.session_state.quiz:
        report_path = generate_pdf(st.session_state.summary, st.session_state.quiz)
        with open(report_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button("📎 Download PDF", data=pdf_bytes, file_name="report.pdf", mime="application/pdf")
    else:
        st.info("📄 Please generate the Summary and Quiz first.")

st.markdown("---")
st.subheader("🛠 Extra Features")
st.session_state.auto_email = st.checkbox("Auto-send summary & quiz to my email")
if st.session_state.auto_email:
    st.session_state.email_to_send = st.text_input("Enter your email for auto-sending:")

st.download_button("💾 Download Chat History", data=export_chat_history(), file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt")

if st.session_state.history:
    with st.expander("🕓 Chat History"):
        for user, bot in reversed(st.session_state.history):
            st.markdown(f"**🧑‍💼 User:** {user}")
            st.markdown(f"**🤖 Bot:** {bot}")
            st.markdown("---")