# AI Tutoring Platform

A web-based AI tutoring platform for South African students, offering personalized math tutoring with multilingual support (English, Afrikaans, Xhosa).

## Project Overview
- **Goal**: Build a free, AI-powered tutoring platform to help students in Cape Town and further.
- **Features (Planned)**:
  - AI chatbot for math tutoring.
  - Aesthetic, user-friendly web interface.
  - Multilingual support for English, Afrikaans, and Xhosa.
- **Tech Stack**: Hugging Face, Streamlit, Tailwind CSS, Firebase, DeepL API.
- **Status**: In development (started June 2025).

## License

This project is licensed under the MIT License.  
See the [`LICENSE`](./LICENSE) file for full terms.

---

**Important Notice**

This project is licensed under the MIT License and is intended **strictly for learning, personal exploration, and demonstration purposes**.

You are welcome to **view and study the code** to understand the implementation and concepts.  
However, **you are *not permitted* to copy, reuse, modify, distribute, or sell** any part of this project, in whole or in part, **without the author’s explicit written permission**.

This project remains the intellectual property of the author and is **not intended for commercial or production use by third parties**.

© 2025 Justin Simon Fussell – All rights reserved.

---

## Setup
- Set up Python 3.13.5 and VS Code on Windows 11 for SAiL development
- This repo uses a `.gitignore` to exclude any sensitive files (e.g., API keys, virtual environments), and ensures the safe handling of all users.

## Progress
**Week 1**
- Successfully created a very basic test Streamlit app to understand the basics.  
- Successfully built a math dataset with 50+ questions and answers for the AI chatbot and tested the AI model in Google Colab’s free cloud platform to ensure correct and accurate answering of each question.

**Week 2**
- Expanded `math_data.csv` to 100+ questions covering arithmetic, fractions, geometry, algebra, trigonometry, probability, and calculus, with step-by-step explanations (e.g., “What is 2 + 2?” → “2 + 2 equals 4”).  
- Developed `app.py` with a hybrid AI system using `sentence-transformers` (`all-MiniLM-L6-v2`) for similarity matching and `distilbert-base-uncased-distilled-squad` for fallback answers, fixing an initial `OSError`.  
- Integrated DeepL API Free (500,000 characters/month) for multilingual support in English, Afrikaans, and Xhosa, with basic error handling.  
- Designed a modern Streamlit UI with custom CSS, a two-column layout (question input/answers and chat history), animated buttons, a gold-themed sidebar, MathJax for math symbols (e.g., x², π), and a branded logo (`assets/SAiL.png`).  
- Added a sidebar for language and subject selection (Math available, Science coming soon) and a math symbols guide.  
- Resolved Git issues (e.g., unrelated histories) and pushed updates to GitHub (`https://github.com/JustinFussell/SAiL.git`).  
- Shared progress on LinkedIn, highlighting features and next steps.
  
- SAiL is a functional AI math tutor with a robust dataset, multilingual capabilities, and a polished UI, ready for local testing.  
- Preparing for Streamlit Cloud deployment with a `requirements.txt` file (`streamlit`, `sentence-transformers`, `transformers`, `torch`, `pandas`, `requests`).  
- Next steps include enhancing error handling (e.g., input validation), adding more dataset variety and possible topic categorization (e.g., word problems and possible dropdowns), and deploying to Streamlit Community Cloud for public access. On top of that is additional run throughs to ensure no errors when interacting with the chatbot regarding math question inputs and answer outputs given by the bot to make sure it does not provide incorrect answers to users. 

## Contact
- Sole Developer: Justin Fussell
- Email: justinfussell23@gmail.com
- LinkedIn: www.linkedin.com/in/justin-fussell
