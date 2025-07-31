# cosmic_scout_ai.py
import os
import streamlit as st
from groq import Groq
from duckduckgo_search import DDGS
from dotenv import load_dotenv
import time
import random
import re
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
import pickle
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import json
import traceback
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Initialize Groq client
api_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None

# Knowledge base files - now in knowledge_base folder
KNOWLEDGE_FILES = {
    "api_docs.txt": "API Documentation",
    "product_info.txt": "Product Information",
    "troubleshooting.txt": "Troubleshooting Guide",
    "code_samples.txt": "Code Samples",
    "assistant_info.txt": "Assistant Information"
}

# Initialize session state
def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm Cosmic Scout 🤖✨ Your AI sidekick with local knowledge! How can I help?", "model": "kimi"}
        ]
    if "model" not in st.session_state:
        st.session_state.model = "kimi"  # Default model
    if "avatar" not in st.session_state:
        st.session_state.avatar = "🤖"  # Default avatar
    if "reactions" not in st.session_state:
        st.session_state.reactions = {}  # Store reactions for each message
    if "selected_emoji" not in st.session_state:
        st.session_state.selected_emoji = None
    if "knowledge_loaded" not in st.session_state:
        st.session_state.knowledge_loaded = False
    if "source_cache" not in st.session_state:
        st.session_state.source_cache = {}
    if "web_search_enabled" not in st.session_state:
        st.session_state.web_search_enabled = True

# Initialize or load knowledge base
def init_knowledge_base():
    if os.path.exists("vectorstore/knowledge_base.index") and os.path.exists("vectorstore/knowledge_meta.pkl"):
        try:
            # Load existing vector store
            index = faiss.read_index("vectorstore/knowledge_base.index")
            with open("vectorstore/knowledge_meta.pkl", "rb") as f:
                metadata = pickle.load(f)
            return index, metadata
        except:
            st.warning("⚠️ Corrupted knowledge base detected. Rebuilding...")
    
    # Create new knowledge base
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    all_chunks = []
    all_metadata = []
    
    for file_name, title in KNOWLEDGE_FILES.items():
        try:
            # Read from knowledge_base folder
            file_path = os.path.join("knowledge_base", file_name)
            with open(file_path, "r") as f:
                content = f.read()
            
            # Split and process text
            chunks = text_splitter.split_text(content)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": file_name,
                    "title": title,
                    "content": chunk[:500] + "..." if len(chunk) > 500 else chunk
                })
        except Exception as e:
            st.error(f"Error loading {file_name}: {str(e)}")
    
    # Create embeddings
    embeddings = embeddings_model.embed_documents(all_chunks)
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    
    # Save to disk
    os.makedirs("vectorstore", exist_ok=True)
    faiss.write_index(index, "vectorstore/knowledge_base.index")
    with open("vectorstore/knowledge_meta.pkl", "wb") as f:
        pickle.dump(all_metadata, f)
    
    return index, all_metadata

# Search knowledge base
def search_knowledge(query, index, metadata, k=3):
    try:
        embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        query_embedding = embeddings_model.embed_query(query)
        query_array = np.array([query_embedding]).astype('float32')
        
        # Perform similarity search
        distances, indices = index.search(query_array, k)
        
        # Format results
        context = "## 📚 LOCAL KNOWLEDGE BASE\n"
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(metadata):
                source = metadata[idx]
                context += (
                    f"### 🔍 {source['title']} ({source['source']})\n"
                    f"{source['content']}\n\n"
                )
        return context
    except Exception as e:
        return f"⚠️ Knowledge search error: {str(e)}"

# Enhanced content extraction with summarization
def extract_main_content(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unnecessary elements
        for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form', 'iframe']):
            element.decompose()
        
        # Get main content elements
        main_content = soup.find('main') or soup.find('article') or soup.body
        
        # Get text and clean
        text = main_content.get_text(separator='\n', strip=True)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Summarize if content is too long
        if len(text) > 2000:
            sentences = text.split('. ')
            if len(sentences) > 10:
                # Take first 3 and last 3 sentences
                summary = '. '.join(sentences[:3] + sentences[-3:])
                return summary
        return text[:2000]  # Return first 2000 characters
    except Exception as e:
        return f"⚠️ Content extraction error: {str(e)}"

# Enhanced web search with content extraction and reliability scoring
def web_search(query, max_results=5, freshness="d"):
    try:
        with DDGS() as ddgs:
            # Get fresh results with time range filter
            results = [r for r in ddgs.text(
                query, 
                max_results=max_results,
                timelimit=freshness
            )]
            
            # Extract and format relevant information
            context = "## 🌐 WEB SEARCH RESULTS\n"
            
            for i, r in enumerate(results[:max_results]):
                # Extract domain for reliability indicator
                domain = urlparse(r['href']).netloc
                
                # Reliability scoring
                reliability_score = 0
                if 'wikipedia.org' in domain:
                    reliability_score = 95
                elif '.edu' in domain or '.gov' in domain:
                    reliability_score = 90
                elif any(d in domain for d in ['nytimes', 'bbc', 'reuters', 'apnews']):
                    reliability_score = 85
                elif any(d in domain for d in ['medium', 'github', 'stackoverflow']):
                    reliability_score = 80
                elif any(d in domain for d in ['blog', 'substack', 'wordpress']):
                    reliability_score = 70
                else:
                    reliability_score = 65
                
                # Date freshness calculation
                freshness_score = "🟢 HIGH"
                freshness_class = "freshness-high"
                if r.get('date'):
                    try:
                        pub_date = datetime.strptime(r['date'], "%Y-%m-%d")
                        days_diff = (datetime.now() - pub_date).days
                        
                        if days_diff > 180:  # 6 months
                            freshness_score = "🔴 LOW"
                            freshness_class = "freshness-low"
                        elif days_diff > 30:  # 1 month
                            freshness_score = "🟡 MEDIUM"
                            freshness_class = "freshness-medium"
                    except:
                        pass
                
                # Extract content snippet
                content_key = f"{r['href']}-{freshness}"
                if content_key in st.session_state.source_cache:
                    content_snippet = st.session_state.source_cache[content_key]
                else:
                    content_snippet = extract_main_content(r['href'])
                    st.session_state.source_cache[content_key] = content_snippet
                
                # Format result with freshness indicator
                context += (
                    f"### 🔍 Result {i+1}: [{r['title']}]({r['href']})\n"
                    f"**Reliability**: {reliability_score}/100 • "
                    f"**Freshness**: <span class='freshness-indicator {freshness_class}'>{freshness_score}</span>\n"
                    f"**Domain**: {domain}\n"
                    f"**Content**: {content_snippet[:300]}...\n\n"
                )
            return context
    except Exception as e:
        return f"⚠️ Web search error: {str(e)}"

# Enhanced RAG response generation with freshness context
def generate_response(query, context, history):
    # Create clean message history
    clean_messages = [
        {"role": "system", "content": f""""
        You are Cosmic Scout, an energetic and knowledgeable AI assistant with access to both local knowledge and web search results.
        Personality: Joyful, creative, and helpful. Use emojis appropriately. 
        Response style: Concise (1-3 sentences) but informative. 
        
        IMPORTANT: Prioritize information based on these rules:
        1. FRESHNESS: For time-sensitive queries, prioritize the most recent information (look for HIGH freshness indicators)
        2. RELIABILITY: When information conflicts, prefer sources with higher reliability scores
        3. LOCAL KNOWLEDGE: For technical questions, prefer local knowledge base over web results
        4. CITATION: Always cite sources using [Source #] notation when possible
        5. CURRENT DATE: Today is {time.strftime('%Y-%m-%d')}
        6. HONESTY: If no relevant information is found, say so honestly
        
        CONTEXT:
        {context}
        """}
    ]
    
    # Add conversation history
    for msg in history:
        if isinstance(msg.get("content"), str):
            clean_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    # Add current query
    clean_messages.append({"role": "user", "content": query})
    
    try:
        completion = client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct",
            messages=clean_messages,
            temperature=0.7, 
            max_tokens=16384,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"🚨 Error: {str(e)}"

# Enhanced reasoning response with date context
def generate_reasoned_response(query, history):
    clean_messages = [
        {"role": "system", "content": f""""
        You are an AI reasoning expert with a passion for clear explanations. 
        Provide detailed, logical explanations with analogies and examples. 
        Use emojis to make complex topics more approachable and joyful! 😊
        Include the current year ({time.strftime('%Y')}) when discussing time-sensitive topics.
        """}
    ]
    
    for msg in history:
        if isinstance(msg.get("content"), str):
            clean_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    clean_messages.append({"role": "user", "content": query})
    
    try:
        completion = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=clean_messages,
            temperature=0.6,
            max_tokens=131072,
            top_p=0.9,
            stream=False,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"🚨 Reasoning Error: {str(e)}"

# Get a random joyful avatar
def get_random_avatar():
    avatars = ["🤖", "✨", "🌟", "🚀", "🌈", "🦄", "🐱", "🐶", "🦊", "🦉", "🦋"]
    return random.choice(avatars)

# Get cosmic theme gradients
def get_theme_gradients():
    return {
        "bg": "linear-gradient(135deg, #0f0c29, #302b63, #24243e)",
        "user": "linear-gradient(90deg, #ff7e5f, #feb47b)",
        "bot": "linear-gradient(90deg, #00c9ff, #92fe9d)",
        "accent": "#ff00cc"
    }

# Process emojis to make them clickable
def make_emojis_clickable(content, message_id):
    # Find all emojis in the content
    emojis = re.findall(r'[\U0001F300-\U0001F9FF]', content)
    unique_emojis = list(set(emojis))
    
    # Replace each emoji with a clickable button
    for emoji in unique_emojis:
        button_id = f"emoji_{message_id}_{emoji}"
        content = content.replace(
            emoji, 
            f'<button class="emoji-btn" data-emoji="{emoji}" data-message="{message_id}" onclick="handleEmojiClick(this)">{emoji}</button>'
        )
    return content

# Determine freshness level based on query
def determine_freshness(query):
    freshness_keywords = {
        "d": ["today", "now", "current", "latest", "recent", "breaking", "update"],
        "w": ["this week", "past 7 days", "recently", "last week"],
        "m": ["this month", "past 30 days", "last month"],
        "y": ["this year", "annual"]
    }
    
    query_lower = query.lower()
    for timeframe, keywords in freshness_keywords.items():
        if any(kw in query_lower for kw in keywords):
            return timeframe
    return "d"  # Default to day

# Main app
def main():
    st.set_page_config(
        page_title="Cosmic Scout AI",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    init_session()
    gradients = get_theme_gradients()
    
    # Initialize knowledge base
    if not st.session_state.knowledge_loaded:
        with st.spinner("🧠 Loading knowledge base..."):
            try:
                knowledge_index, knowledge_meta = init_knowledge_base()
                st.session_state.knowledge_index = knowledge_index
                st.session_state.knowledge_meta = knowledge_meta
                st.session_state.knowledge_loaded = True
            except Exception as e:
                st.error(f"Failed to load knowledge base: {str(e)}")
                st.session_state.knowledge_loaded = False
    
    # Custom CSS with cosmic theme and effects
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@700&family=Poppins:wght@500;800&family=Fredoka:wght@700&display=swap');
    
    body, .stApp {{
        background: {gradients['bg']};
        background-attachment: fixed;
        min-height: 100vh;
        overflow-x: hidden;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }}
    
    /* Floating particles */
    .particle {{
        position: fixed;
        pointer-events: none;
        border-radius: 50%;
        z-index: -1;
    }}
    
    .stChatFloatingInputContainer {{ 
        background: rgba(255,255,255,0.95)!important; 
        border-radius: 25px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        padding: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        max-width: 800px;
        margin: 0 auto;
    }}
    
    .title {{ 
        font-family: 'Fredoka', sans-serif; 
        font-weight: 700;
        color: white; 
        text-align: center;
        font-size: 4.5rem;
        text-shadow: 0 0 20px {gradients['accent']}, 0 0 30px rgba(255,255,255,0.7);
        margin-bottom: 0;
        letter-spacing: 1px;
        background: linear-gradient(45deg, #f3ec78, #af4261);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: glow 3s infinite alternate;
    }}
    
    @keyframes glow {{
        from {{ text-shadow: 0 0 10px {gradients['accent']}, 0 0 20px rgba(255,255,255,0.7); }}
        to {{ text-shadow: 0 0 30px {gradients['accent']}, 0 0 40px rgba(255,255,255,0.9); }}
    }}
    
    .subheader {{ 
        color: #fffd82 !important; 
        text-align: center;
        font-family: 'Comic Neue', cursive;
        font-size: 1.6rem;
        margin-top: 0;
        margin-bottom: 2rem;
        text-shadow: 0 0 10px rgba(255,255,255,0.5);
        animation: float 6s ease-in-out infinite;
    }}
    
    @keyframes float {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-15px); }}
        100% {{ transform: translateY(0px); }}
    }}
    
    .st-emotion-cache-1kyxreq {{ 
        justify-content: center; 
    }}
    
    .message-container {{
        border-radius: 25px!important;
        padding: 20px!important;
        margin-bottom: 15px!important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255,255,255,0.1);
        transform-style: preserve-3d;
    }}
    
    .message-container::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255,255,255,0.05);
        z-index: -1;
        border-radius: 25px;
    }}
    
    .message-container:hover {{
        transform: translateY(-8px) rotateX(5deg) rotateY(5deg);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }}
    
    .user-message {{
        background: {gradients['user']}!important;
        border-left: 5px solid rgba(255,255,255,0.3);
    }}
    
    .bot-message {{
        background: {gradients['bot']}!important;
        border-left: 5px solid rgba(255,255,255,0.3);
    }}
    
    .model-tag {{
        position: absolute;
        bottom: 10px;
        right: 15px;
        font-size: 0.7rem;
        background: rgba(0,0,0,0.3);
        color: white;
        padding: 3px 12px;
        border-radius: 15px;
        font-weight: bold;
        backdrop-filter: blur(5px);
    }}
    
    .stSpinner > div {{
        background: {gradients['accent']} !important;
    }}
    
    .stButton button {{
        background: rgba(255,255,255,0.2) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        border-radius: 25px !important;
        font-weight: bold !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: all 0.3s ease !important;
        font-family: 'Comic Neue', cursive;
        backdrop-filter: blur(5px);
    }}
    
    .stButton button:hover {{
        transform: scale(1.05);
        box-shadow: 0 8px 25px {gradients['accent']};
        background: rgba(255,255,255,0.3) !important;
    }}
    
    .stRadio > div {{
        background: rgba(255,255,255,0.15);
        border-radius: 15px;
        padding: 15px;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255,255,255,0.1);
    }}
    
    .stRadio label {{
        color: white !important;
        font-weight: bold;
        font-family: 'Comic Neue', cursive;
        text-shadow: 0 0 5px rgba(255,255,255,0.5);
    }}
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        color: white !important;
        font-family: 'Fredoka', sans-serif;
        text-shadow: 0 0 10px rgba(255,255,255,0.5);
    }}
    
    .chat-history {{
        height: 500px;
        width: 800px;
        overflow-y: auto;
        padding: 20px;
        border-radius: 25px;
        background: rgba(255,255,255,0.1);
        box-shadow: inset 0 8px 32px rgba(0,0,0,0.2);
        margin: 0 auto 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        display: flex;
        flex-direction: column;
    }}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {{
        width: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {gradients['accent']};
        border-radius: 10px;
    }}
    
    /* Animation for new messages */
    @keyframes slideIn {{
        from {{ transform: translateY(20px); opacity: 0; }}
        to {{ transform: translateY(0); opacity: 1; }}
    }}
    
    @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.05); }}
        100% {{ transform: scale(1); }}
    }}
    
    .new-message {{
        animation: slideIn 0.4s ease-out, pulse 1s 0.4s;
    }}
    
    .avatar-container {{
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
        perspective: 1000px;
    }}
    
    .avatar {{
        font-size: 5rem;
        animation: bounce 2s infinite, rotate 15s infinite linear;
        text-shadow: 0 0 20px {gradients['accent']};
        transform-style: preserve-3d;
    }}
    
    @keyframes bounce {{
        0%, 100% {{ transform: translateY(0) rotateY(0); }}
        50% {{ transform: translateY(-20px) rotateY(180deg); }}
    }}
    
    @keyframes rotate {{
        0% {{ transform: rotateY(0); }}
        100% {{ transform: rotateY(360deg); }}
    }}
    
    .confetti {{
        position: absolute;
        width: 20px;
        height: 20px;
        font-size: 20px;
        line-height: 1;
        text-align: center;
        animation: confetti-fall 5s linear;
        z-index: 9999;
    }}
    
    @keyframes confetti-fall {{
        0% {{ transform: translateY(-100px) rotate(0deg); opacity: 1; }}
        100% {{ transform: translateY(100vh) rotate(720deg); opacity: 0; }}
    }}
    
    .fun-fact {{
        background: rgba(255,255,255,0.15);
        border-radius: 15px;
        padding: 15px;
        margin: 20px 0;
        text-align: center;
        font-family: 'Comic Neue', cursive;
        color: white;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255,255,255,0.1);
    }}
    
    .search-preview {{
        background: rgba(255,255,255,0.15);
        border-radius: 15px;
        padding: 15px;
        margin-top: 10px;
        font-size: 0.9rem;
        max-height: 200px;
        overflow-y: auto;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255,255,255,0.1);
    }}
    
    .reaction-bar {{
        display: flex;
        justify-content: flex-end;
        gap: 5px;
        margin-top: 10px;
    }}
    
    .reaction-btn {{
        background: rgba(0,0,0,0.2);
        border: none;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        font-size: 16px;
        cursor: pointer;
        transition: all 0.3s;
        backdrop-filter: blur(5px);
    }}
    
    .reaction-btn:hover {{
        transform: scale(1.3);
        background: rgba(255,255,255,0.3);
    }}
    
    .reaction-btn.active {{
        transform: scale(1.3);
        background: rgba(255,255,255,0.4);
        box-shadow: 0 0 10px {gradients['accent']};
    }}
    
    .parallax-container {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
    }}
    
    .parallax-layer {{
        position: absolute;
        width: 100%;
        height: 100%;
    }}
    
    .stars {{
        background: url('https://www.transparenttextures.com/patterns/stardust.png');
        opacity: 0.3;
        animation: stars 120s linear infinite;
    }}
    
    @keyframes stars {{
        0% {{ background-position: 0 0; }}
        100% {{ background-position: 1000px 1000px; }}
    }}
    
    .twinkling {{
        background: url('https://www.transparenttextures.com/patterns/starfield.png');
        opacity: 0.4;
        animation: twinkling 200s linear infinite;
    }}
    
    @keyframes twinkling {{
        0% {{ background-position: 0 0; }}
        100% {{ background-position: -1000px 500px; }}
    }}
    
    .floating-island {{
        position: absolute;
        width: 300px;
        height: 300px;
        bottom: -150px;
        right: -150px;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 70%);
        border-radius: 50%;
        filter: blur(20px);
        animation: floatIsland 25s ease-in-out infinite;
    }}
    
    @keyframes floatIsland {{
        0% {{ transform: translate(0, 0) rotate(0deg); }}
        25% {{ transform: translate(-50px, -50px) rotate(5deg); }}
        50% {{ transform: translate(0, -100px) rotate(0deg); }}
        75% {{ transform: translate(50px, -50px) rotate(-5deg); }}
        100% {{ transform: translate(0, 0) rotate(0deg); }}
    }}
    
    .floating-island:nth-child(2) {{
        width: 200px;
        height: 200px;
        top: 20%;
        left: -100px;
        animation: floatIsland 20s ease-in-out infinite reverse;
    }}
    
    .emoji-btn {{
        background: none;
        border: none;
        font-size: 1.2em;
        cursor: pointer;
        padding: 2px;
        transition: all 0.3s;
        transform: scale(1);
    }}
    
    .emoji-btn:hover {{
        transform: scale(1.3);
        text-shadow: 0 0 10px {gradients['accent']};
    }}
    
    .emoji-btn.active {{
        transform: scale(1.4);
        text-shadow: 0 0 15px {gradients['accent']};
        animation: pulse 0.5s infinite alternate;
    }}
    
    .knowledge-badge {{
        position: absolute;
        top: 10px;
        right: 10px;
        background: rgba(0,0,0,0.3);
        color: white;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: bold;
        backdrop-filter: blur(5px);
    }}
    
    .freshness-indicator {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: bold;
        margin-left: 5px;
    }}
    
    .freshness-high {{
        background: rgba(76, 175, 80, 0.3);
        border: 1px solid #4CAF50;
    }}
    
    .freshness-medium {{
        background: rgba(255, 193, 7, 0.3);
        border: 1px solid #FFC107;
    }}
    
    .freshness-low {{
        background: rgba(244, 67, 54, 0.3);
        border: 1px solid #F44336;
    }}
    
    .source-badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: bold;
        margin: 5px 0;
        background: rgba(33, 150, 243, 0.2);
        border: 1px solid #2196F3;
    }}
    
    .reliability-meter {{
        display: inline-block;
        width: 60px;
        height: 10px;
        background: #555;
        border-radius: 5px;
        overflow: hidden;
        vertical-align: middle;
        margin-left: 5px;
    }}
    
    .reliability-fill {{
        height: 100%;
        background: linear-gradient(90deg, #f44336, #ffc107, #4caf50);
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Add cosmic background effects
    st.markdown("""
    <div class="parallax-container">
        <div class="parallax-layer stars"></div>
        <div class="parallax-layer twinkling"></div>
        <div class="floating-island"></div>
        <div class="floating-island"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add confetti effect for fun
    if st.session_state.get("show_confetti", False):
        st.markdown("""
        <script>
        function createConfetti() {
            const container = document.querySelector('.stApp');
            const emojis = ['🎉', '✨', '🌟', '🚀', '🌈', '🦄', '🐱', '🐶', '🦊', '🦉', '🦋'];
            
            for (let i = 0; i < 100; i++) {
                const confetti = document.createElement('div');
                confetti.className = 'confetti';
                confetti.innerText = emojis[Math.floor(Math.random() * emojis.length)];
                confetti.style.left = Math.random() * 100 + 'vw';
                confetti.style.fontSize = Math.random() * 20 + 20 + 'px';
                confetti.style.animationDuration = Math.random() * 3 + 2 + 's';
                confetti.style.color = ['#ff9a9e', '#a1c4fd', '#fad0c4', '#c2e9fb', '#ffd700', '#ff6b6b'][Math.floor(Math.random() * 6)];
                container.appendChild(confetti);
                
                setTimeout(() => {
                    confetti.remove();
                }, 5000);
            }
        }
        
        setTimeout(createConfetti, 100);
        </script>
    """, unsafe_allow_html=True)
        st.session_state.show_confetti = False
    
    # Header with enhanced design
    st.markdown('<div class="avatar-container"><div class="avatar">' + st.session_state.avatar + '</div></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="title">COSMIC SCOUT</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Chat • Search • Reason • Learn • Play</p>', unsafe_allow_html=True)
    
    # Knowledge base status
    kb_status = "✅ Loaded" if st.session_state.knowledge_loaded else "❌ Failed"
    st.caption(f"🧠 Knowledge Base: {len(st.session_state.knowledge_meta) if st.session_state.knowledge_loaded else 0} documents | {kb_status}")
    
    # Fixed square chat box
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-history">', unsafe_allow_html=True)
        
        # Display messages with animations
        for i, msg in enumerate(st.session_state.messages):
            avatar = st.session_state.avatar if msg["role"] == "assistant" else "👤"
            css_class = "bot-message" if msg["role"] == "assistant" else "user-message"
            anim_class = "new-message" if i == len(st.session_state.messages)-1 else ""
            
            with st.chat_message(msg["role"], avatar=avatar):
                content = msg["content"]
                model_tag = f'<span class="model-tag">{msg.get("model", "").upper()}</span>' if msg.get("model") else ""
                
                # Make emojis clickable only for assistant messages
                if msg["role"] == "assistant":
                    message_id = f"msg_{i}"
                    content = make_emojis_clickable(content, message_id)
                
                st.markdown(
                    f'<div class="message-container {css_class} {anim_class}">'
                    f'{content}'
                    f'{model_tag}'
                    f'</div>', 
                    unsafe_allow_html=True
                )
                
                # Add reaction bar for assistant messages
                if msg["role"] == "assistant":
                    message_id = id(msg)
                    reactions = st.session_state.reactions.get(message_id, [])
                    
                    st.markdown("""
                    <div class="reaction-bar">
                        <button class="reaction-btn" onclick="handleReaction(this, '👍')">👍</button>
                        <button class="reaction-btn" onclick="handleReaction(this, '❤️')">❤️</button>
                        <button class="reaction-btn" onclick="handleReaction(this, '😂')">😂</button>
                        <button class="reaction-btn" onclick="handleReaction(this, '😮')">😮</button>
                        <button class="reaction-btn" onclick="handleReaction(this, '👏')">👏</button>
                    </div>
                    """, unsafe_allow_html=True)
                
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Text input
    text_input = st.chat_input("Ask me anything...")
    
    # Generate response
    if text_input:
        st.session_state.messages.append({"role": "user", "content": text_input})
        
        # Determine which model to use
        use_reasoning = st.session_state.model == "reasoning" or any(keyword in text_input.lower() for keyword in ["explain", "why", "how", "reason", "detail"])
        
        if use_reasoning:
            with st.spinner("🧠 Deep thinking..."):
                response = generate_reasoned_response(text_input, st.session_state.messages)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "model": "reasoning"
                })
        else:
            # Search knowledge base
            kb_context = ""
            if st.session_state.knowledge_loaded:
                with st.spinner("🔍 Consulting knowledge base..."):
                    kb_context = search_knowledge(
                        text_input, 
                        st.session_state.knowledge_index, 
                        st.session_state.knowledge_meta
                    )
            
            # Determine freshness level based on query
            freshness = determine_freshness(text_input)
            
            web_context = ""
            if st.session_state.web_search_enabled:
                # Perform web search with determined freshness
                with st.spinner(f"🌐 Searching web ({freshness} freshness)..."):
                    web_context = web_search(text_input, freshness=freshness)
            
            # Combine contexts
            full_context = f"{kb_context}\n\n{web_context}"
            
            # Show context preview
            with st.expander("🔍 Context Sources Preview", expanded=True):
                st.markdown(f'<div class="search-preview">{full_context}</div>', unsafe_allow_html=True)
            
            with st.spinner("🤖 Crafting response..."):
                response = generate_response(text_input, full_context, st.session_state.messages)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "model": "kimi"
                })
        
        # Change avatar for fun
        st.session_state.avatar = get_random_avatar()
        st.session_state.show_confetti = True
        st.rerun()
    
    # Sidebar settings with model selection
    with st.sidebar:
        st.markdown("## ⚙️ COSMIC SETTINGS")
        st.caption(f"🧠 Available Models: Kimi & DeepSeek-R1")
        
        # Model selection
        st.session_state.model = st.radio(
            "Select AI Mode:",
            ["kimi", "reasoning"],
            format_func=lambda x: "🤖 Quick Chat" if x == "kimi" else "🧠 Deep Reasoning",
            index=0 if st.session_state.model == "kimi" else 1
        )
        
        # Web search toggle
        st.session_state.web_search_enabled = st.checkbox(
            "🌐 Enable Web Search", 
            value=st.session_state.web_search_enabled,
            help="Disable to only use local knowledge base"
        )
        
        # Knowledge base info
        st.markdown("## 📚 KNOWLEDGE BASE")
        if st.session_state.knowledge_loaded:
            st.success("Knowledge base loaded successfully!")
            st.caption(f"Documents: {len(st.session_state.knowledge_meta)}")
            
            # Show document list
            doc_counts = {}
            for meta in st.session_state.knowledge_meta:
                doc_counts[meta["source"]] = doc_counts.get(meta["source"], 0) + 1
                
            for file, count in doc_counts.items():
                st.caption(f"• {KNOWLEDGE_FILES.get(file, file)}: {count} chunks")
        else:
            st.error("Knowledge base failed to load")
            
        if st.button("🔄 Rebuild Knowledge Base", use_container_width=True):
            st.session_state.knowledge_loaded = False
            st.rerun()
        
        # Fun facts
        fun_facts = [
            "I combine local knowledge with web search for comprehensive answers",
            "Pro tip: Use words like 'latest' for fresh web results",
            "The knowledge base contains technical docs and API info",
            "Deep reasoning mode is great for complex explanations",
            "Try asking about Groq API or pricing information",
            "The cosmic background is made of pure stardust",
            "My avatar changes with every conversation!",
            "Click on emojis to see them float across the screen",
            "The floating islands are powered by cosmic energy"
        ]
        st.markdown(f'<div class="fun-fact">✨ {random.choice(fun_facts)} ✨</div>', unsafe_allow_html=True)
        
        # Spacer between fun fact and avatar button
        st.markdown("<div style='margin: 25px 0;'></div>", unsafe_allow_html=True)
        
        if st.button("🎉 Change My Avatar", use_container_width=True):
            st.session_state.avatar = get_random_avatar()
            st.session_state.show_confetti = True
            st.rerun()
            
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": "Chat cleared! What can I help with now? 😊", "model": "kimi"}
            ]
            st.session_state.avatar = get_random_avatar()
            st.session_state.show_confetti = True
            st.rerun()
        
        st.markdown("---")
        st.markdown("### CHAT HISTORY")
        st.caption(f"{len(st.session_state.messages)} messages")
        
        # Export chat history
        if st.button("💾 Export Chat", use_container_width=True):
            chat_text = "\n\n".join([
                f"{st.session_state.avatar if msg['role']=='assistant' else '👤'} {msg['content']}" 
                for msg in st.session_state.messages
            ])
            st.download_button(
                label="Download Chat",
                data=chat_text,
                file_name=f"cosmic_scout_chat_{time.strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # JavaScript for interactive effects
    st.markdown("""
    <script>
    // Add floating particles on mouse move
    document.addEventListener('mousemove', function(e) {
        const particles = document.createElement('div');
        particles.className = 'particle';
        particles.style.left = e.pageX + 'px';
        particles.style.top = e.pageY + 'px';
        particles.style.width = Math.random() * 15 + 5 + 'px';
        particles.style.height = particles.style.width;
        particles.style.background = `hsl(${Math.random() * 360}, 70%, 60%)`;
        particles.style.opacity = Math.random();
        particles.style.animation = `float ${Math.random() * 6 + 4}s linear forwards`;
        
        document.body.appendChild(particles);
        
        setTimeout(() => {
            particles.remove();
        }, 5000);
    });
    
    // Handle reaction clicks
    function handleReaction(button, emoji) {
        // Remove active class from all buttons
        const buttons = button.parentElement.querySelectorAll('.reaction-btn');
        buttons.forEach(btn => btn.classList.remove('active'));
        
        // Add active class to clicked button
        button.classList.add('active');
        
        // Create floating reaction
        const reaction = document.createElement('div');
        reaction.innerText = emoji;
        reaction.style.position = 'absolute';
        reaction.style.left = (button.getBoundingClientRect().left + window.scrollX) + 'px';
        reaction.style.top = (button.getBoundingClientRect().top + window.scrollY) + 'px';
        reaction.style.fontSize = '24px';
        reaction.style.zIndex = '9999';
        reaction.style.animation = 'floatUp 2s forwards';
        reaction.style.pointerEvents = 'none';
        
        document.body.appendChild(reaction);
        
        // Define the floatUp animation
        const style = document.createElement('style');
        style.innerHTML = `
            @keyframes floatUp {
                0% { transform: translateY(0) scale(1); opacity: 1; }
                100% { transform: translateY(-100px) scale(2); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
        
        // Remove after animation
        setTimeout(() => {
            reaction.remove();
            style.remove();
        }, 2000);
    }
    
    // Handle emoji clicks in messages
    function handleEmojiClick(button) {
        const emoji = button.getAttribute('data-emoji');
        const messageId = button.getAttribute('data-message');
        
        // Create floating emoji
        const floatingEmoji = document.createElement('div');
        floatingEmoji.innerText = emoji;
        floatingEmoji.style.position = 'fixed';
        floatingEmoji.style.left = (button.getBoundingClientRect().left + window.scrollX) + 'px';
        floatingEmoji.style.top = (button.getBoundingClientRect().top + window.scrollY) + 'px';
        floatingEmoji.style.fontSize = '24px';
        floatingEmoji.style.zIndex = '9999';
        floatingEmoji.style.animation = 'floatEmoji 3s forwards';
        floatingEmoji.style.pointerEvents = 'none';
        
        document.body.appendChild(floatingEmoji);
        
        // Add animation style
        const style = document.createElement('style');
        style.innerHTML = `
            @keyframes floatEmoji {
                0% { 
                    transform: translate(0, 0) scale(1); 
                    opacity: 1;
                }
                100% { 
                    transform: translate(${Math.random() * 400 - 200}px, -200px) scale(3); 
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
        
        // Remove after animation
        setTimeout(() => {
            floatingEmoji.remove();
            style.remove();
        }, 3000);
        
        // Add visual feedback to clicked emoji
        button.classList.add('active');
        setTimeout(() => {
            button.classList.remove('active');
        }, 500);
    }
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    if not api_key:
        st.error("🚫 GROQ_API_KEY not found in environment variables!")
    else:
        main()