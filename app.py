# ============================================================
#   app.py — Streamlit UI for Intelligent QA System
#   NLP Final Project
# ============================================================

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"   # Force PyTorch-only, skip TF/Keras imports

import streamlit as st
import time
import re
import string
import numpy as np
import pandas as pd

# ── Page configuration ──────────────────────────────────────
st.set_page_config(
    page_title="Intelligent QA System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for clean, minimal styling ────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .main {
        background-color: #0f1117;
    }

    /* Title styling */
    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #8a8fa8;
        margin-bottom: 2rem;
    }

    /* Answer card */
    .answer-card {
        background: linear-gradient(135deg, #1a1f35, #242840);
        border: 1px solid #3d4266;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-top: 1rem;
    }

    .answer-text {
        font-size: 1.15rem;
        font-weight: 600;
        color: #a78bfa;
    }

    /* Context card */
    .context-card {
        background: #161b2e;
        border: 1px solid #2d3354;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-top: 0.8rem;
        font-size: 0.92rem;
        color: #b0b8d4;
        line-height: 1.7;
    }

    /* Metric badges */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }

    .metric-badge {
        background: #1e2440;
        border: 1px solid #2e3660;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 0.85rem;
        color: #7c8db5;
    }

    .metric-badge span {
        color: #818cf8;
        font-weight: 600;
    }

    /* Sidebar styling */
    .sidebar-header {
        font-size: 1rem;
        font-weight: 600;
        color: #c4c9e2;
        margin-bottom: 0.5rem;
        margin-top: 1rem;
    }

    .pipeline-step {
        padding: 0.3rem 0;
        font-size: 0.88rem;
        color: #8a90b0;
    }

    /* Sample question chip */
    .stButton > button {
        background: #1e2440;
        border: 1px solid #3d4580;
        color: #a0a8cc;
        border-radius: 8px;
        font-size: 0.85rem;
        padding: 0.3rem 0.8rem;
        transition: all 0.2s;
        width: 100%;
        text-align: left;
    }

    .stButton > button:hover {
        background: #2a3060;
        border-color: #6366f1;
        color: #c4c9ff;
    }

    /* Answer button */
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border: none;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.6rem 2rem;
        border-radius: 10px;
        width: auto;
    }

    div[data-testid="stButton"] > button[kind="primary"]:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    .divider {
        border-top: 1px solid #2a2f4a;
        margin: 1.5rem 0;
    }

    .score-bar-label {
        font-size: 0.82rem;
        color: #6b7280;
        margin-bottom: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
#   DATA — Built-in knowledge base (SQuAD-inspired contexts)
#   These passages cover ML/AI topics for reliable demo results
# ============================================================

KNOWLEDGE_BASE = [
    {
        "id": 1,
        "topic": "Machine Learning",
        "context": (
            "Machine learning is a subfield of artificial intelligence that gives computers the ability "
            "to learn from data without being explicitly programmed. It focuses on developing algorithms "
            "that can access data and use it to learn for themselves. Machine learning models are trained "
            "on large datasets and can make predictions or decisions by identifying patterns. Common types "
            "include supervised learning, unsupervised learning, and reinforcement learning. Supervised "
            "learning uses labeled training data to learn a mapping from inputs to outputs. Applications "
            "of machine learning include image recognition, natural language processing, recommendation "
            "systems, fraud detection, and autonomous vehicles."
        ),
    },
    {
        "id": 2,
        "topic": "Overfitting",
        "context": (
            "Overfitting is a modeling error in machine learning that occurs when a model learns the "
            "training data too well, including its noise and random fluctuations. An overfitted model "
            "has poor generalization ability, meaning it performs well on training data but fails on "
            "unseen test data. Overfitting happens when the model is too complex relative to the amount "
            "of training data available. Techniques to prevent overfitting include regularization "
            "(L1 and L2), dropout layers in neural networks, cross-validation, early stopping, pruning "
            "decision trees, and collecting more training data. The bias-variance tradeoff describes "
            "the balance between underfitting (high bias) and overfitting (high variance)."
        ),
    },
    {
        "id": 3,
        "topic": "Neural Networks",
        "context": (
            "A neural network is a computational model inspired by the structure and function of the "
            "human brain. It consists of layers of interconnected nodes called neurons or perceptrons. "
            "A typical neural network has an input layer, one or more hidden layers, and an output layer. "
            "Each connection between neurons has a weight that is adjusted during training using an "
            "optimization algorithm called backpropagation combined with gradient descent. Neural networks "
            "can approximate any continuous function and are used in deep learning for tasks such as "
            "image classification, speech recognition, language translation, and game playing."
        ),
    },
    {
        "id": 4,
        "topic": "Natural Language Processing",
        "context": (
            "Natural Language Processing (NLP) is a branch of artificial intelligence concerned with "
            "the interaction between computers and human language. NLP involves tasks such as text "
            "classification, named entity recognition, machine translation, question answering, sentiment "
            "analysis, and text summarization. The NLP pipeline typically includes tokenization, "
            "stopword removal, stemming or lemmatization, part-of-speech tagging, and parsing. "
            "Modern NLP systems are powered by transformer-based models like BERT, GPT, and RoBERTa, "
            "which use self-attention mechanisms to capture long-range dependencies in text."
        ),
    },
    {
        "id": 5,
        "topic": "Deep Learning",
        "context": (
            "Deep learning is a subset of machine learning that uses artificial neural networks with "
            "many layers (deep architectures) to learn representations of data with multiple levels of "
            "abstraction. Deep learning has achieved remarkable results in computer vision, speech "
            "recognition, and natural language processing. Key architectures include Convolutional "
            "Neural Networks (CNNs) for image data, Recurrent Neural Networks (RNNs) and LSTMs for "
            "sequential data, and Transformers for language tasks. Deep learning models require large "
            "amounts of data and computational resources, typically using GPUs for training."
        ),
    },
    {
        "id": 6,
        "topic": "Transformer and BERT",
        "context": (
            "The Transformer is a deep learning architecture introduced in the paper 'Attention is All "
            "You Need' by Vaswani et al. in 2017. It relies entirely on a self-attention mechanism to "
            "compute representations of input and output without using recurrent or convolutional layers. "
            "BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained Transformer "
            "model developed by Google. BERT is trained using masked language modeling and next sentence "
            "prediction objectives. It achieves state-of-the-art results on many NLP benchmarks including "
            "question answering, text classification, and named entity recognition."
        ),
    },
    {
        "id": 7,
        "topic": "TF-IDF",
        "context": (
            "TF-IDF stands for Term Frequency-Inverse Document Frequency. It is a numerical statistic "
            "used in information retrieval and text mining to reflect how important a word is to a document "
            "in a collection. Term Frequency (TF) measures how often a word appears in a document. "
            "Inverse Document Frequency (IDF) measures how rare or common a word is across all documents. "
            "Words that appear frequently in a document but rarely across all documents receive high "
            "TF-IDF scores. TF-IDF is widely used for document ranking, keyword extraction, and as "
            "features in text classification models."
        ),
    },
    {
        "id": 8,
        "topic": "Cross-Validation",
        "context": (
            "Cross-validation is a statistical technique used to evaluate machine learning models on "
            "limited data. In k-fold cross-validation, the dataset is divided into k equal-sized folds. "
            "The model is trained on k-1 folds and validated on the remaining fold, repeating k times "
            "so each fold is used as the validation set exactly once. The average performance across "
            "all folds gives an unbiased estimate of model performance. Common variants include "
            "Stratified k-fold (preserves class proportions) and Leave-One-Out CV. Cross-validation "
            "is essential for hyperparameter tuning and model selection without overfitting to the test set."
        ),
    },
    {
        "id": 9,
        "topic": "Support Vector Machine",
        "context": (
            "A Support Vector Machine (SVM) is a supervised machine learning algorithm used for "
            "classification and regression tasks. SVM finds the optimal hyperplane that maximally "
            "separates different classes in the feature space. The data points closest to the hyperplane "
            "are called support vectors. SVMs can handle non-linearly separable data using the kernel "
            "trick, which maps data into higher-dimensional spaces where linear separation is possible. "
            "Common kernels include linear, polynomial, and radial basis function (RBF). SVMs are "
            "effective in high-dimensional spaces and are used in text classification, image recognition, "
            "and bioinformatics."
        ),
    },
    {
        "id": 10,
        "topic": "Gradient Descent",
        "context": (
            "Gradient descent is an iterative optimization algorithm used to minimize a loss function "
            "in machine learning. The algorithm computes the gradient (partial derivative) of the loss "
            "with respect to each model parameter and updates parameters in the opposite direction of "
            "the gradient. The learning rate controls the step size of each update. Variants include "
            "Batch Gradient Descent (uses all data), Stochastic Gradient Descent (uses one sample), "
            "and Mini-batch Gradient Descent (uses a subset). Adaptive optimizers like Adam, RMSProp, "
            "and Adagrad automatically adjust learning rates for each parameter during training."
        ),
    },
    {
        "id": 11,
        "topic": "Regularization",
        "context": (
            "Regularization is a set of techniques used to prevent overfitting in machine learning models "
            "by adding a penalty term to the loss function. L1 regularization (Lasso) adds the absolute "
            "value of coefficients to the loss, which can drive some coefficients to zero, performing "
            "feature selection. L2 regularization (Ridge) adds the squared magnitude of coefficients, "
            "which shrinks coefficients toward zero but rarely eliminates them. Elastic Net combines "
            "both L1 and L2 penalties. In neural networks, dropout is a powerful regularization technique "
            "that randomly sets a fraction of neurons to zero during training to prevent co-adaptation."
        ),
    },
    {
        "id": 12,
        "topic": "Decision Trees and Random Forests",
        "context": (
            "A decision tree is a supervised learning algorithm that splits data based on feature values "
            "to make predictions. Each internal node represents a feature, each branch represents a "
            "decision rule, and each leaf node represents a predicted outcome. Random Forest is an "
            "ensemble method that builds multiple decision trees and combines their predictions through "
            "majority voting (classification) or averaging (regression). Random forests are robust to "
            "overfitting and can handle missing values. Feature importance scores from random forests "
            "help identify the most influential variables in a dataset, making them useful for "
            "exploratory data analysis."
        ),
    },
]

CONTEXTS = [item["context"] for item in KNOWLEDGE_BASE]
TOPICS = [item["topic"] for item in KNOWLEDGE_BASE]


# ============================================================
#   CORE NLP FUNCTIONS
# ============================================================

def clean_text(text: str) -> str:
    """Remove special characters and normalize whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s.,!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Simple stopword list (no NLTK dependency needed for app)
STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for", "of",
    "and", "or", "but", "with", "by", "from", "as", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "this",
    "that", "these", "those", "i", "you", "he", "she", "we", "they",
    "what", "which", "who", "how", "when", "where", "why", "all", "any",
    "their", "its", "there", "here", "not", "no", "so", "up", "out",
    "also", "about", "into", "through", "during", "before", "after"
}


def tokenize(text: str) -> list[str]:
    """Simple word tokenizer."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [t for t in text.split() if t]


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS]


# ============================================================
#   TF-IDF ENGINE (built from scratch — no sklearn dependency)
# ============================================================

def build_tfidf_index(corpus: list[str]):
    """Build TF-IDF index over a list of documents."""
    import math

    tokenized = [remove_stopwords(tokenize(doc)) for doc in corpus]
    vocab = sorted(set(tok for doc in tokenized for tok in doc))
    word2idx = {w: i for i, w in enumerate(vocab)}
    N = len(corpus)

    # Document frequency
    df = np.zeros(len(vocab))
    for doc_tokens in tokenized:
        unique_tokens = set(doc_tokens)
        for tok in unique_tokens:
            if tok in word2idx:
                df[word2idx[tok]] += 1

    # IDF
    idf = np.log((N + 1) / (df + 1)) + 1.0

    # TF-IDF matrix
    tfidf_matrix = np.zeros((N, len(vocab)))
    for doc_idx, doc_tokens in enumerate(tokenized):
        tf = {}
        for tok in doc_tokens:
            tf[tok] = tf.get(tok, 0) + 1
        for tok, count in tf.items():
            if tok in word2idx:
                j = word2idx[tok]
                tfidf_matrix[doc_idx, j] = (count / len(doc_tokens)) * idf[j]

    # L2-normalize rows
    norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    tfidf_matrix = tfidf_matrix / norms

    return vocab, word2idx, idf, tfidf_matrix


def query_tfidf(question: str, vocab, word2idx, idf, tfidf_matrix, top_k=3):
    """Retrieve top-k relevant contexts using cosine similarity."""
    q_tokens = remove_stopwords(tokenize(question))
    q_vec = np.zeros(len(vocab))
    for tok in q_tokens:
        if tok in word2idx:
            j = word2idx[tok]
            q_vec[j] = idf[j]

    norm = np.linalg.norm(q_vec)
    if norm == 0:
        return [], []
    q_vec /= norm

    scores = tfidf_matrix @ q_vec
    top_indices = np.argsort(scores)[::-1][:top_k]
    return top_indices.tolist(), scores[top_indices].tolist()


# ============================================================
#   EXTRACTIVE QA (sentence scoring)
# ============================================================

def extractive_answer(question: str, context: str) -> str:
    """Score sentences by keyword overlap and return best sentence."""
    q_tokens = set(remove_stopwords(tokenize(question)))
    sentences = re.split(r'(?<=[.!?])\s+', context)

    best_score = -1
    best_sentence = sentences[0] if sentences else context

    for sent in sentences:
        s_tokens = set(remove_stopwords(tokenize(sent)))
        if not s_tokens:
            continue
        overlap = len(q_tokens & s_tokens)
        score = overlap / (len(q_tokens) + 0.1)
        if score > best_score:
            best_score = score
            best_sentence = sent

    return best_sentence.strip()


# ============================================================
#   BERT-based QA (HuggingFace — loaded lazily)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_bert_qa_pipeline():
    """Load HuggingFace QA pipeline once and cache."""
    try:
        from transformers import pipeline
        return pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            tokenizer="deepset/roberta-base-squad2",
            framework="pt",   # Force PyTorch — avoids Keras 3 conflict
        )
    except Exception as e:
        return None


@st.cache_resource(show_spinner=False)
def load_tfidf_index():
    """Build and cache TF-IDF index over the knowledge base."""
    return build_tfidf_index(CONTEXTS)


# ============================================================
#   SIDEBAR
# ============================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🤖 Intelligent QA System")
        st.markdown("*NLP Final Project*")
        st.divider()

        st.markdown("### 📘 About")
        st.markdown(
            "This system answers questions using two methods:\n\n"
            "- **TF-IDF + Extractive**: Fast, classical NLP\n"
            "- **BERT QA**: Deep learning, higher accuracy\n\n"
            "Built on a curated ML/AI knowledge base."
        )
        st.divider()

        st.markdown("### 🔬 NLP Pipeline Steps")
        steps = [
            "1️⃣  Text Cleaning",
            "2️⃣  Tokenization",
            "3️⃣  Normalization",
            "4️⃣  Stopword Removal",
            "5️⃣  Stemming / Lemmatization",
            "6️⃣  POS Tagging",
            "7️⃣  Dependency Parsing",
            "8️⃣  Feature Engineering",
            "9️⃣  Information Retrieval",
            "🔟 Question Answering",
        ]
        for step in steps:
            st.markdown(f"<div class='pipeline-step'>{step}</div>", unsafe_allow_html=True)

        st.divider()

        st.markdown("### 💡 Sample Questions")
        sample_questions = [
            "What is machine learning?",
            "Explain overfitting",
            "What is a neural network?",
            "How does gradient descent work?",
            "What is TF-IDF?",
            "How does dropout prevent overfitting?",
            "What is BERT?",
            "Explain cross-validation",
            "What is a support vector machine?",
            "How does random forest work?",
        ]

        if "question_input" not in st.session_state:
            st.session_state["question_input"] = ""

        for q in sample_questions:
            if st.button(f"💬 {q}", key=f"sample_{q}"):
                st.session_state["question_input"] = q
                st.rerun()


# ============================================================
#   MAIN PAGE
# ============================================================

def main():
    render_sidebar()

    # ── Header ──────────────────────────────────────────────
    st.markdown('<div class="main-title">🤖 Intelligent QA System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">End-to-End NLP Pipeline · Information Retrieval · Question Answering</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Input row ───────────────────────────────────────────
    col_input, col_method = st.columns([3, 1])

    with col_method:
        method = st.selectbox(
            "⚙️ Method",
            options=["TF-IDF Extractive (Fast)", "BERT QA (Accurate)"],
            help="TF-IDF is instant. BERT loads on first use (~20–30s).",
        )

    with col_input:
        question = st.text_input(
            "🔍 Ask your question",
            value=st.session_state.get("question_input", ""),
            placeholder="e.g. What is machine learning? Explain overfitting...",
            key="question_box",
            label_visibility="visible",
        )
        # Sync state
        st.session_state["question_input"] = question

    col_btn, col_spacer = st.columns([1, 5])
    with col_btn:
        run = st.button("Get Answer ▶", type="primary")

    # ── Results ─────────────────────────────────────────────
    if run and question.strip():
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Build TF-IDF index (cached)
        with st.spinner("🔍 Retrieving relevant context..."):
            vocab, word2idx, idf, tfidf_matrix = load_tfidf_index()
            top_indices, top_scores = query_tfidf(
                question, vocab, word2idx, idf, tfidf_matrix, top_k=3
            )

        if not top_indices:
            st.warning("⚠️ Could not find relevant context. Please rephrase your question.")
            return

        best_idx = top_indices[0]
        best_context = CONTEXTS[best_idx]
        best_topic = TOPICS[best_idx]
        best_score = top_scores[0]

        # ── Generate answer ──────────────────────────────────
        start_time = time.time()

        if "BERT" in method:
            with st.spinner("🤖 Loading BERT model (first load ~30s, then instant)..."):
                qa_pipe = load_bert_qa_pipeline()

            if qa_pipe is None:
                st.error(
                    "❌ BERT model could not be loaded (likely missing `transformers` or `torch`).\n\n"
                    "Install with: `pip install transformers torch`\n\n"
                    "Falling back to TF-IDF Extractive method."
                )
                answer = extractive_answer(question, best_context)
                method_used = "TF-IDF Extractive (fallback)"
            else:
                result = qa_pipe(question=question, context=best_context)
                answer = result["answer"]
                confidence = result["score"]
                method_used = "BERT QA (RoBERTa-base-SQuAD2)"
        else:
            answer = extractive_answer(question, best_context)
            method_used = "TF-IDF + Extractive Sentence Scoring"

        elapsed = time.time() - start_time

        # ── Display results ──────────────────────────────────
        st.markdown("### 💡 Answer")
        st.markdown(
            f'<div class="answer-card"><span class="answer-text">{answer}</span></div>',
            unsafe_allow_html=True,
        )

        # Metrics row
        confidence_display = f"{confidence:.2%}" if ("BERT" in method and qa_pipe is not None) else "N/A"
        st.markdown(
            f"""
            <div class="metric-row">
                <div class="metric-badge">⚙️ Method: <span>{method_used}</span></div>
                <div class="metric-badge">📌 Topic: <span>{best_topic}</span></div>
                <div class="metric-badge">📊 Similarity: <span>{best_score:.3f}</span></div>
                {"<div class='metric-badge'>🎯 Confidence: <span>" + confidence_display + "</span></div>" if confidence_display != "N/A" else ""}
                <div class="metric-badge">⏱️ Time: <span>{elapsed:.2f}s</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Retrieved context
        st.markdown("### 📄 Retrieved Context")
        st.markdown(
            f'<div class="context-card">{best_context}</div>',
            unsafe_allow_html=True,
        )

        # Top-K retrieved passages (expandable)
        if len(top_indices) > 1:
            with st.expander(f"📚 View all {len(top_indices)} retrieved passages with similarity scores"):
                for rank, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
                    st.markdown(f"**Rank #{rank} — {TOPICS[idx]}** (score: `{score:.4f}`)")
                    st.markdown(
                        f'<div class="context-card" style="margin-bottom:1rem;">{CONTEXTS[idx]}</div>',
                        unsafe_allow_html=True,
                    )

        # NLP preprocessing preview
        with st.expander("🔬 NLP Preprocessing Preview (what happened under the hood)"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Question**")
                st.code(question, language=None)
                st.markdown("**After Cleaning**")
                st.code(clean_text(question), language=None)
                tokens = tokenize(question)
                st.markdown("**Tokens**")
                st.code(str(tokens), language=None)
            with col2:
                filtered = remove_stopwords(tokens)
                st.markdown("**After Stopword Removal**")
                st.code(str(filtered), language=None)
                st.markdown("**Query Terms Used for Retrieval**")
                st.code(str(filtered), language=None)
                st.markdown("**Matched Topic**")
                st.success(f"✅ {best_topic}")

    elif run and not question.strip():
        st.warning("⚠️ Please enter a question before clicking **Get Answer**.")

    # ── Footer ───────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        "<center><small style='color:#4a5068;'>NLP Final Project · Intelligent QA System · "
        "Built with NLTK · scikit-learn · HuggingFace Transformers · Streamlit</small></center>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
