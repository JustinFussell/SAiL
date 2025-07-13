# Import necessary libraries for Streamlit, data handling, AI models, and API requests
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch
import requests

# Configure the page with a wide layout, title, and icon
st.set_page_config(page_title="SAiL: South African Intelligent Learning", page_icon=":books:", layout="wide")

# Apply custom CSS for a modern, interactive design and include MathJax for math rendering
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stTextInput > div > div > input {
        border: 2px solid #00A86B;
        border-radius: 12px;
        padding: 12px;
        font-size: 16px;
        transition: border-color 0.3s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #005AFF;
        box-shadow: 0 0 8px rgba(0, 90, 255, 0.3);
    }
    .stButton > button {
        background-color: #00A86B;
        color: white;
        border-radius: 12px;
        padding: 12px 24px;
        font-size: 16px;
        border: none;
        transition: background-color 0.3s ease, transform 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #005AFF;
        transform: scale(1.05);
    }
    h1, h2, h3 {color: #005AFF; font-family: 'Montserrat', sans-serif;}
    .sidebar .sidebar-content {background-color: #FFD700; padding: 10px;}
    .chat-message {background-color: #e6f3ff; border-radius: 10px; padding: 10px; margin: 5px 0;}
    .chat-question {font-weight: bold; color: #005AFF;}
    .chat-answer {color: #333;}
    </style>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
""", unsafe_allow_html=True)

# Display header with logo and introductory text
st.image("assets/SAiL.png", width=200)
st.title("SAiL: South African Intelligent Learning")
st.markdown("Your free AI-powered math tutor for students in Cape Town and beyond!")

# Create sidebar for user settings including language, topic, and subject selection
st.sidebar.header("Settings")
language = st.sidebar.selectbox("Select Language", ["English", "Afrikaans", "Xhosa"])
topic = st.sidebar.selectbox("Select Math Topic", ["All", "Arithmetic", "Algebra", "Geometry", "Trigonometry", "Calculus", "Probability"])
subject = st.sidebar.selectbox("Select Subject", ["Math (Available)", "Science (Coming Soon)"])

# Initialize session state to store chat history across interactions
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load and cache the math dataset for efficient reuse
@st.cache_data
def load_data():
    return pd.read_csv("data/math_data.csv")

data = load_data()

# Load and cache the sentence transformer model for question similarity matching
@st.cache_resource
def load_similarity_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

similarity_model = load_similarity_model()

# Load and cache the question-answering model as a fallback
@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

chatbot = load_qa_model()

# Set up DeepL API for text translation
DEEPL_API_KEY = "0bb5521b-b76b-458c-9f42-7922c24500a2:fx"  
DEEPL_URL = "https://api-free.deepl.com/v2/translate"

# Function to translate text using DeepL API with fallback to original text
def translate_text(text, target_lang):
    if target_lang == "English":
        return text
    lang_map = {"Afrikaans": "AF", "Xhosa": "XH"}
    response = requests.post(
        DEEPL_URL,
        data={
            "auth_key": DEEPL_API_KEY,
            "text": text,
            "target_lang": lang_map[target_lang]
        }
    )
    if response.status_code == 200:
        return response.json()["translations"][0]["text"]
    st.warning("Translation failed, using original text.")
    return text

# Set up a two-column layout for question input and chat history
col1, col2 = st.columns([2, 1])

with col1:
    st.header(f"Ask a Math Question ({language})")
    question = st.text_input(f"Enter your math question in {language} (e.g., What is 2 + 2?):")
    if st.button("Get Answer"):
        if not question.strip():
            st.error("Please enter a valid question.")
        else:
            # Translate the question to English for processing
            question_en = translate_text(question, "English") if language != "English" else question
            
            # Filter dataset based on selected topic
            filtered_data = data if topic == "All" else data[data["topic"] == topic]
            
            # Compute similarity between user question and filtered dataset questions
            question_embedding = similarity_model.encode(question_en, convert_to_tensor=True)
            dataset_questions = filtered_data["question"].tolist()
            dataset_embeddings = similarity_model.encode(dataset_questions, convert_to_tensor=True)
            similarities = util.cos_sim(question_embedding, dataset_embeddings)[0]
            
            # Determine the best answer based on similarity or fallback to QA model
            max_similarity = similarities.max().item()
            if max_similarity > 0.7:  # Adjusted threshold to 0.7 for stricter matching
                best_match_idx = similarities.argmax().item()
                answer = filtered_data["answer"].iloc[best_match_idx]
            else:
                context = " ".join(filtered_data["answer"].tolist())
                result = chatbot(question=question_en, context=context)
                answer = result['answer']
            
            # Translate the answer to the selected language
            answer_translated = translate_text(answer, language)
            
            # Format answer with MathJax for proper rendering of math symbols
            answer_formatted = answer_translated.replace("x^2", "x<sup>2</sup>").replace("x^3", "x<sup>3</sup>").replace("π", "π")
            st.session_state.chat_history.append((question, answer_formatted))
            
            st.write(f"**Answer**: {answer_formatted}", unsafe_allow_html=True)

with col2:
    st.header("Chat History")
    # Display the chat history with styled messages
    for q, a in st.session_state.chat_history:
        st.markdown(f'<div class="chat-message"><span class="chat-question">{q}</span><br><span class="chat-answer">{a}</span></div>', unsafe_allow_html=True)

# Provide a reference guide for math symbols
st.markdown("### Math Symbols Reference")
st.markdown("""
Use these symbols in your questions:
- Addition: +
- Subtraction: −
- Multiplication: × or *
- Division: ÷ or /
- Exponents: ^ (e.g., x^2 for x²)
- Pi: π
""", unsafe_allow_html=True)

# Add a footer with credits and links
st.markdown("---")
st.markdown("Built by Justin Fussell | Powered by Streamlit & Hugging Face | [GitHub](https://github.com/JustinFussell/SAiL.git) | [LinkedIn](https://linkedin.com/in/justin-fussell)")