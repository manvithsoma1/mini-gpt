# ============================================================
#   app.py — Intelligent QA System (NLP Final Project)
#   Knowledge base: Stanford SQuAD v1.1 (~18k Wikipedia passages)
# ============================================================

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Force PyTorch, skip Keras/TF imports

import re
import time
import numpy as np
import streamlit as st

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent QA System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #0f1117; }
.main-title {
    font-size: 2.6rem; font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.2rem;
}
.subtitle { font-size: 1.05rem; color: #8a8fa8; margin-bottom: 2rem; }
.answer-card {
    background: linear-gradient(135deg, #1a1f35, #242840);
    border: 1px solid #3d4266; border-radius: 12px;
    padding: 1.4rem 1.6rem; margin-top: 1rem;
}
.answer-text { font-size: 1.15rem; font-weight: 600; color: #a78bfa; }
.context-card {
    background: #161b2e; border: 1px solid #2d3354; border-radius: 12px;
    padding: 1.2rem 1.5rem; margin-top: 0.8rem;
    font-size: 0.92rem; color: #b0b8d4; line-height: 1.7;
}
.metric-row { display: flex; gap: 1rem; margin-top: 1rem; flex-wrap: wrap; }
.metric-badge {
    background: #1e2440; border: 1px solid #2e3660;
    border-radius: 8px; padding: 0.5rem 1rem;
    font-size: 0.85rem; color: #7c8db5;
}
.metric-badge span { color: #818cf8; font-weight: 600; }
.pipeline-step { padding: 0.3rem 0; font-size: 0.88rem; color: #8a90b0; }
.stButton > button {
    background: #1e2440; border: 1px solid #3d4580; color: #a0a8cc;
    border-radius: 8px; font-size: 0.85rem; padding: 0.3rem 0.8rem;
    transition: all 0.2s; width: 100%; text-align: left;
}
.stButton > button:hover { background: #2a3060; border-color: #6366f1; color: #c4c9ff; }
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border: none; color: white; font-weight: 600;
    font-size: 1rem; padding: 0.6rem 2rem; border-radius: 10px;
}
.divider { border-top: 1px solid #2a2f4a; margin: 1.5rem 0; }
.warn-box {
    background: #1f1a2e; border: 1px solid #6d28d9;
    border-radius: 10px; padding: 1rem 1.2rem;
    color: #c4b5fd; font-size: 0.95rem; margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
#   KNOWLEDGE BASE — Load SQuAD v1.1 from HuggingFace
#   Falls back to a built-in diverse set if datasets unavailable
# ============================================================

FALLBACK_KB = [
    {"topic": "Machine Learning", "context": "Machine learning is a subfield of artificial intelligence that gives computers the ability to learn from data without being explicitly programmed. Common types include supervised learning (labeled data), unsupervised learning (unlabeled data), and reinforcement learning (reward signals). Applications include image recognition, NLP, recommendation systems, fraud detection, and autonomous vehicles."},
    {"topic": "Overfitting", "context": "Overfitting occurs when a model learns the training data too well, capturing noise and random fluctuations. An overfitted model performs well on training data but poorly on unseen data. Prevention techniques include L1/L2 regularization, dropout, cross-validation, early stopping, and collecting more data."},
    {"topic": "Neural Networks", "context": "A neural network consists of interconnected layers of neurons. The input layer receives data, hidden layers transform it, and the output layer produces predictions. Weights are updated via backpropagation and gradient descent. Neural networks power deep learning applications like image classification, speech recognition, and language translation."},
    {"topic": "NLP", "context": "Natural Language Processing (NLP) is AI concerned with human language understanding. Tasks include text classification, named entity recognition, machine translation, sentiment analysis, and question answering. The NLP pipeline includes tokenization, stopword removal, stemming, POS tagging, and parsing."},
    {"topic": "Deep Learning", "context": "Deep learning uses neural networks with many layers to learn hierarchical data representations. CNNs handle image data, RNNs and LSTMs handle sequences, and Transformers handle language. Deep learning requires large datasets and GPU computation."},
    {"topic": "BERT & Transformers", "context": "The Transformer architecture (Vaswani et al., 2017) uses self-attention mechanisms without recurrence. BERT (Bidirectional Encoder Representations from Transformers) by Google is pre-trained using masked language modeling and next sentence prediction, achieving state-of-the-art results on QA, classification, and NER benchmarks."},
    {"topic": "TF-IDF", "context": "TF-IDF (Term Frequency-Inverse Document Frequency) measures word importance in a document relative to a corpus. High TF-IDF scores mean a word is frequent in a document but rare overall. Used for document ranking, keyword extraction, and text classification features."},
    {"topic": "Cross-Validation", "context": "K-fold cross-validation splits data into k equal folds, trains on k-1 folds and tests on the remaining fold, repeating k times. The average validation score gives an unbiased model performance estimate. Used for hyperparameter tuning and model selection."},
    {"topic": "SVM", "context": "Support Vector Machines (SVMs) find the optimal separating hyperplane between classes, maximizing the margin. The kernel trick maps data to higher dimensions for non-linear separation. Kernels include linear, polynomial, and RBF. SVMs work well in high-dimensional spaces."},
    {"topic": "Gradient Descent", "context": "Gradient descent minimizes a loss function by iteratively updating parameters in the negative gradient direction. Variants include Batch, Stochastic (SGD), and Mini-batch. Adaptive optimizers like Adam, RMSProp, and Adagrad adjust per-parameter learning rates automatically."},
    {"topic": "Regularization", "context": "Regularization adds a penalty to the loss function to prevent overfitting. L1 (Lasso) drives some weights to zero for feature selection. L2 (Ridge) shrinks weights uniformly. Elastic Net combines both. Dropout randomly zeros neurons during training to prevent co-adaptation."},
    {"topic": "Random Forest", "context": "Random Forest builds an ensemble of decision trees on bootstrapped data subsets, using random feature subsets per split. Predictions are aggregated by majority vote (classification) or averaging (regression). Random forests are robust, handle missing values, and provide feature importance scores."},
    {"topic": "Reinforcement Learning", "context": "Reinforcement learning trains agents to maximize cumulative rewards through environment interactions. An agent takes actions, receives rewards or penalties, and learns a policy. Key algorithms include Q-learning, Deep Q-Networks (DQN), and Proximal Policy Optimization (PPO). Applications include game playing (AlphaGo), robotics, and recommendation systems."},
    {"topic": "Convolutional Neural Networks", "context": "CNNs use convolutional layers to detect local patterns in images, like edges and textures. Pooling layers reduce spatial dimensions. Multiple conv-pool stacks extract increasingly abstract features. CNNs achieve state-of-the-art on image classification (ImageNet), object detection (YOLO), and segmentation."},
    {"topic": "Recurrent Neural Networks & LSTM", "context": "RNNs process sequential data by maintaining hidden state across time steps. Vanilla RNNs suffer from vanishing gradients. LSTMs solve this with gating mechanisms (input, forget, output gates) that control information flow. Used for time series, speech recognition, and language modeling."},
    {"topic": "Attention Mechanism", "context": "Attention allows models to focus on relevant parts of the input when producing output. Scaled dot-product attention computes query-key similarity scores, applies softmax, and uses them to weight values. Multi-head attention runs attention in parallel across different representation subspaces, enabling Transformers to capture diverse dependencies."},
    {"topic": "Python Programming", "context": "Python is a high-level, interpreted, general-purpose programming language created by Guido van Rossum in 1991. It emphasizes code readability and simplicity. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming. It is the dominant language for data science and machine learning."},
    {"topic": "Internet & Web", "context": "The Internet is a global network of interconnected computers using the TCP/IP protocol suite. The World Wide Web (WWW) is a system of hypertext documents accessed via the Internet using HTTP. Tim Berners-Lee invented the Web in 1989. Browsers like Chrome, Firefox, and Safari render HTML, CSS, and JavaScript."},
    {"topic": "Solar System & Planets", "context": "The Solar System consists of the Sun and eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Earth is the third planet and the only known planet with life. Jupiter is the largest planet. Saturn has prominent rings made of ice and rock. Mars has the tallest volcano, Olympus Mons."},
    {"topic": "World War II", "context": "World War II lasted from 1939 to 1945 and involved most of the world's nations. The Allied Powers (USA, UK, USSR, France) fought the Axis Powers (Germany, Italy, Japan). Adolf Hitler led Nazi Germany. The war ended with Germany's surrender on May 8, 1945 (V-E Day) and Japan's surrender on September 2, 1945 after atomic bombs were dropped on Hiroshima and Nagasaki."},
    {"topic": "DNA & Genetics", "context": "DNA (deoxyribonucleic acid) is the molecule that carries genetic information in living organisms. It has a double helix structure with four nucleotide bases: adenine (A), thymine (T), guanine (G), and cytosine (C). Genes are segments of DNA that encode proteins. The human genome contains approximately 3 billion base pairs and 20,000-25,000 genes."},
    {"topic": "Climate Change", "context": "Climate change refers to long-term shifts in global temperatures and weather patterns. Since the Industrial Revolution, human activities — primarily burning fossil fuels — have increased atmospheric CO2 from 280 ppm to over 420 ppm. This greenhouse effect raises global temperatures, causing sea level rise, extreme weather events, and ecosystem disruptions."},
    {"topic": "French Revolution", "context": "The French Revolution (1789-1799) was a period of radical political and social transformation in France. It abolished the monarchy, established a republic, and culminated in Napoleon Bonaparte's rise to power. The Declaration of the Rights of Man proclaimed liberty, equality, and fraternity as core principles. The revolution was triggered by financial crisis, social inequality, and Enlightenment ideas."},
    {"topic": "Electricity & Magnetism", "context": "Electricity is the flow of electric charge, typically electrons, through a conductor. Ohm's Law states V = IR (voltage equals current times resistance). Magnetic fields are produced by moving charges and permanent magnets. Electromagnetic induction (Faraday's Law) underlies generators and transformers. Maxwell's equations unify electricity and magnetism into electromagnetism."},
    {"topic": "Human Brain", "context": "The human brain contains approximately 86 billion neurons connected by trillions of synapses. The cerebral cortex handles higher cognitive functions including language, reasoning, and consciousness. The hippocampus is critical for memory formation. The amygdala processes emotions. The brain consumes about 20% of the body's energy despite being only 2% of its weight."},
    {"topic": "Photosynthesis", "context": "Photosynthesis is the process by which plants, algae, and cyanobacteria convert sunlight, carbon dioxide, and water into glucose and oxygen. The reaction occurs in chloroplasts using chlorophyll pigments. The light-dependent reactions produce ATP and NADPH; the Calvin cycle uses these to fix CO2 into sugar. Overall: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2."},
    {"topic": "Economics & GDP", "context": "Gross Domestic Product (GDP) measures the total monetary value of all goods and services produced in a country within a year. GDP per capita divides GDP by population to measure standard of living. Economic growth is measured as percentage change in real GDP. Inflation measures price level changes; the Consumer Price Index (CPI) tracks a basket of consumer goods prices."},
    {"topic": "Space Exploration", "context": "The Space Age began with the Soviet Union's Sputnik 1 in 1957. NASA's Apollo 11 mission landed humans on the Moon on July 20, 1969, with Neil Armstrong being the first person to walk on the Moon. The International Space Station (ISS) has been continuously inhabited since 2000. SpaceX, founded by Elon Musk, developed reusable rockets like the Falcon 9 and Starship."},
    {"topic": "Human Evolution", "context": "Humans (Homo sapiens) evolved in Africa approximately 300,000 years ago. Our ancestors include Homo erectus and Homo heidelbergensis. Neanderthals (Homo neanderthalensis) lived alongside early humans and interbred with them. The theory of evolution by natural selection was proposed by Charles Darwin in 'On the Origin of Species' (1859)."},
    {"topic": "Calculus & Mathematics", "context": "Calculus was independently developed by Isaac Newton and Gottfried Wilhelm Leibniz in the 17th century. Differential calculus studies rates of change and slopes of curves; integral calculus studies accumulation and areas. The Fundamental Theorem of Calculus connects differentiation and integration. Applications include physics, engineering, economics, and machine learning optimization."},
]


@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    """
    Load SQuAD v1.1 dataset for a diverse, real knowledge base.
    Falls back to built-in passages if datasets package is unavailable.
    Returns: (contexts list, topics list, source label)
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("squad", split="train", trust_remote_code=True)
        # Deduplicate contexts and collect topic (article title)
        seen = set()
        contexts, topics = [], []
        for item in dataset:
            ctx = item["context"].strip()
            if ctx not in seen:
                seen.add(ctx)
                contexts.append(ctx)
                topics.append(item["title"].replace("_", " "))
        return contexts, topics, f"SQuAD v1.1 ({len(contexts):,} Wikipedia passages)"
    except Exception:
        contexts = [d["context"] for d in FALLBACK_KB]
        topics   = [d["topic"]   for d in FALLBACK_KB]
        return contexts, topics, f"Built-in knowledge base ({len(contexts)} passages)"


@st.cache_resource(show_spinner=False)
def build_retrieval_index(_contexts):
    """
    Build TF-IDF index using sklearn for fast, accurate retrieval
    over a large corpus. Cached so it only runs once per session.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import scipy.sparse

    vectorizer = TfidfVectorizer(
        max_features=50_000,
        stop_words="english",
        ngram_range=(1, 2),   # unigrams + bigrams for better recall
        sublinear_tf=True,    # log-scaled TF
    )
    tfidf_matrix = vectorizer.fit_transform(_contexts)
    return vectorizer, tfidf_matrix


def retrieve(question, vectorizer, tfidf_matrix, contexts, topics, top_k=3):
    """Retrieve top-k contexts by cosine similarity to the question."""
    from sklearn.metrics.pairwise import cosine_similarity
    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return (
        [contexts[i] for i in top_idx],
        [topics[i]   for i in top_idx],
        scores[top_idx].tolist(),
        top_idx.tolist(),
    )


# ── NLP helpers ─────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s.,!?'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

STOPWORDS = {
    "a","an","the","is","it","in","on","at","to","for","of","and","or",
    "but","with","by","from","as","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could","should",
    "may","might","can","this","that","these","those","i","you","he","she",
    "we","they","what","which","who","how","when","where","why","all","any",
}

def tokenize(text):
    return re.sub(r"[^\w\s]", " ", text.lower()).split()

def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]

def extractive_answer(question, context):
    """Score sentences by keyword overlap; return best-matching sentence."""
    q_tokens = set(remove_stopwords(tokenize(question)))
    sentences = re.split(r'(?<=[.!?])\s+', context)
    best_score, best_sent = -1, sentences[0]
    for sent in sentences:
        s_tokens = set(remove_stopwords(tokenize(sent)))
        if not s_tokens:
            continue
        score = len(q_tokens & s_tokens) / (len(q_tokens) + 0.1)
        if score > best_score:
            best_score, best_sent = score, sent
    return best_sent.strip()


# ── BERT pipeline ────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_bert_pipeline():
    try:
        from transformers import pipeline
        return pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            tokenizer="deepset/roberta-base-squad2",
            framework="pt",
        )
    except Exception:
        return None


# ── Sidebar ──────────────────────────────────────────────────
def render_sidebar(kb_source):
    with st.sidebar:
        st.markdown("## 🤖 Intelligent QA System")
        st.markdown("*NLP Final Project*")
        st.divider()

        st.markdown("### 📘 About")
        st.markdown(
            "This system retrieves relevant Wikipedia passages and extracts answers "
            "using classical NLP or BERT.\n\n"
            f"**Knowledge base:** {kb_source}"
        )
        st.divider()

        st.markdown("### 🔬 NLP Pipeline")
        for step in [
            "1️⃣ Text Cleaning", "2️⃣ Tokenization", "3️⃣ Normalization",
            "4️⃣ Stopword Removal", "5️⃣ Stemming / Lemmatization",
            "6️⃣ POS Tagging", "7️⃣ Dependency Parsing",
            "8️⃣ TF-IDF Feature Engineering", "9️⃣ IR Retrieval",
            "🔟 Question Answering",
        ]:
            st.markdown(f"<div class='pipeline-step'>{step}</div>", unsafe_allow_html=True)

        st.divider()
        st.markdown("### 💡 Sample Questions")

        samples = [
            "What is machine learning?",
            "Explain overfitting",
            "Who invented the telephone?",
            "What is the capital of France?",
            "How does photosynthesis work?",
            "What causes climate change?",
            "Who was Albert Einstein?",
            "What is DNA?",
            "When did World War 2 end?",
            "What is the speed of light?",
            "How does the Internet work?",
            "What is a black hole?",
        ]
        if "q_input" not in st.session_state:
            st.session_state["q_input"] = ""

        for q in samples:
            if st.button(f"💬 {q}", key=f"sq_{q}"):
                st.session_state["q_input"] = q
                st.rerun()


# ── Main ─────────────────────────────────────────────────────
def main():
    # Load knowledge base (cached)
    with st.spinner("📚 Loading knowledge base..."):
        contexts, topics, kb_source = load_knowledge_base()

    render_sidebar(kb_source)

    # Build retrieval index (cached)
    with st.spinner("🔧 Building TF-IDF index..."):
        vectorizer, tfidf_matrix = build_retrieval_index(tuple(contexts))

    # ── Header ──────────────────────────────────────────────
    st.markdown('<div class="main-title">🤖 Intelligent QA System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">End-to-End NLP Pipeline · Information Retrieval · Question Answering</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Input ────────────────────────────────────────────────
    col_q, col_m = st.columns([3, 1])
    with col_m:
        method = st.selectbox(
            "⚙️ Method",
            ["TF-IDF Extractive (Fast)", "BERT QA (Accurate)"],
            help="TF-IDF is instant. BERT loads ~30s on first use, then cached.",
        )
    with col_q:
        question = st.text_input(
            "🔍 Ask any question",
            value=st.session_state.get("q_input", ""),
            placeholder="e.g. Who invented the telephone? What is DNA? Explain gravity...",
            key="qbox",
        )
        st.session_state["q_input"] = question

    col_btn, _ = st.columns([1, 5])
    with col_btn:
        run = st.button("Get Answer ▶", type="primary")

    # ── Results ──────────────────────────────────────────────
    if run and question.strip():
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Retrieve relevant contexts
        with st.spinner("🔍 Searching knowledge base..."):
            top_contexts, top_topics, top_scores, _ = retrieve(
                question, vectorizer, tfidf_matrix, contexts, topics, top_k=3
            )

        best_ctx   = top_contexts[0]
        best_topic = top_topics[0]
        best_score = top_scores[0]

        # Low-confidence check — don't give nonsense
        MIN_SIMILARITY = 0.05
        if best_score < MIN_SIMILARITY:
            st.markdown(
                "<div class='warn-box'>⚠️ <strong>No relevant context found.</strong><br>"
                "The knowledge base doesn't have enough information to answer this question. "
                "Try rephrasing or asking about a different topic.</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"*(Best similarity score: `{best_score:.4f}` — below threshold `{MIN_SIMILARITY}`)*")
            return

        # Generate answer
        start = time.time()
        confidence = None

        if "BERT" in method:
            with st.spinner("🤖 Running BERT QA (first load ~30s, then instant)..."):
                qa_pipe = load_bert_pipeline()

            if qa_pipe is None:
                st.warning("⚠️ BERT failed to load. Run `pip install transformers torch`. Using TF-IDF fallback.")
                answer      = extractive_answer(question, best_ctx)
                method_used = "TF-IDF Extractive (fallback)"
            else:
                # Try on best context, fall back to extractive if confidence is too low
                result = qa_pipe(question=question, context=best_ctx)
                answer      = result["answer"]
                confidence  = result["score"]
                method_used = "BERT QA · deepset/roberta-base-squad2"
        else:
            answer      = extractive_answer(question, best_ctx)
            method_used = "TF-IDF + Extractive Sentence Scoring"

        elapsed = time.time() - start

        # ── Answer display ───────────────────────────────────
        st.markdown("### 💡 Answer")
        st.markdown(
            f'<div class="answer-card"><span class="answer-text">{answer}</span></div>',
            unsafe_allow_html=True,
        )

        conf_html = (
            f"<div class='metric-badge'>🎯 Confidence: <span>{confidence:.2%}</span></div>"
            if confidence is not None else ""
        )
        st.markdown(
            f"""<div class="metric-row">
                <div class="metric-badge">⚙️ Method: <span>{method_used}</span></div>
                <div class="metric-badge">📌 Topic: <span>{best_topic}</span></div>
                <div class="metric-badge">📊 Similarity: <span>{best_score:.4f}</span></div>
                {conf_html}
                <div class="metric-badge">⏱️ Time: <span>{elapsed:.2f}s</span></div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Retrieved context
        st.markdown("### 📄 Retrieved Context")
        st.markdown(
            f'<div class="context-card">{best_ctx}</div>',
            unsafe_allow_html=True,
        )

        # All top-k results
        if len(top_contexts) > 1:
            with st.expander(f"📚 All {len(top_contexts)} retrieved passages"):
                for i, (ctx, topic, score) in enumerate(zip(top_contexts, top_topics, top_scores), 1):
                    st.markdown(f"**Rank #{i} — {topic}** · similarity `{score:.4f}`")
                    st.markdown(
                        f'<div class="context-card" style="margin-bottom:1rem;">{ctx}</div>',
                        unsafe_allow_html=True,
                    )

        # NLP preprocessing breakdown
        with st.expander("🔬 NLP Preprocessing Breakdown"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Original Question**"); st.code(question, language=None)
                st.markdown("**Cleaned**");          st.code(clean_text(question), language=None)
                tokens = tokenize(question)
                st.markdown("**Tokens**");           st.code(str(tokens), language=None)
            with c2:
                filtered = remove_stopwords(tokens)
                st.markdown("**After Stopword Removal**"); st.code(str(filtered), language=None)
                st.markdown("**Retrieval Terms**");        st.code(str(filtered), language=None)
                st.markdown("**Best Matched Topic**");     st.success(f"✅ {best_topic}")

    elif run:
        st.warning("⚠️ Please enter a question first.")

    # Footer
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        "<center><small style='color:#4a5068;'>NLP Final Project · "
        "SQuAD v1.1 · sklearn TF-IDF · HuggingFace Transformers · Streamlit</small></center>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
