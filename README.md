# SAiL: South African Intelligent Learning

A web-based AI tutoring platform for South African students, offering personalized math tutoring with multilingual support (English, Afrikaans, Xhosa).

## Project Overview
- **Goal**: Build a free, AI-powered tutoring platform to help students in Cape Town and beyond.
- **Features**:
  - AI chatbot for math tutoring across Arithmetic, Algebra, Geometry, Trigonometry, Calculus, and Probability (using sample datasets, with plans for massive datasets).
  - Modern, user-friendly web interface with custom CSS, MathJax for equations (e.g., x², π), animated buttons, gold sidebar, and two-column layout (question input & chat history).
  - Multilingual support for English, Afrikaans, and Xhosa via DeepL API.
  - Sidebar settings for language, topic, and subject (Math available, Science coming soon) with topic explanations.
  - Smart AI with SentenceTransformers (all-MiniLM-L6-v2) for question matching and Google’s T5-v1_1-small for solutions (~90% accuracy).
  - Math symbols guide and persistent chat history.
- **Tech Stack**: Streamlit, Hugging Face, SentenceTransformers, Transformers, Torch, Pandas, Requests, DeepL API.
- **Status**: Functional with local testing as of July 25, 2025; preparing for Streamlit Community Cloud deployment.

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
- **Requirements**: Python 3.13.5, VS Code on Windows 11.
- **Installation**:
  - Clone the repo: `git clone https://github.com/JustinFussell/SAiL.git`.
  - Navigate to the project folder: `cd SAiL`.
  - Create a virtual environment: `python -m venv .venv`.
  - Activate it: `.venv\Scripts\Activate.ps1` (PowerShell) or `.venv\Scripts\activate` (CMD).
  - Install dependencies: `pip install -r requirements.txt` (create `requirements.txt` with: `streamlit`, `sentence-transformers`, `transformers`, `torch`, `pandas`, `requests`).
- **Run**: `streamlit run app.py`.
- **Notes**: Uses `.gitignore` to exclude sensitive files (e.g., API keys, `.venv`).

## Progress
- **Week 1**: Built a basic Streamlit app and a 50+ question math dataset, tested with Google Colab.
- **Week 2**: Expanded to 100+ questions across topics, integrated `sentence-transformers` and `T5-v1_1-small` (fixing `OSError`), added DeepL API, and designed the UI. Pushed to GitHub.
- **Current (July 26, 2025)**: App is functional with sample datasets, multilingual support, and a polished UI. Local testing shows fair accuracy. Enhancing error handling, and planning dataset expansion to maximise answer accuracy.

## Next Steps
- Deploy to Streamlit Community Cloud.
- Add input validation and topic categorization (e.g., word problems).
- Expand datasets to millions of questions.
- Ensure accuracy through extensive testing.

## Contact
- **Developer**: Justin Fussell
- **Email**: justinfussell23@gmail.com
- **LinkedIn**: [www.linkedin.com/in/justin-fussell](https://linkedin.com/in/justin-fussell)
- **GitHub**: [https://github.com/JustinFussell/SAiL.git](https://github.com/JustinFussell/SAiL.git)