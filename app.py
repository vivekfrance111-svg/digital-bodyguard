import streamlit as st
import torch
import time
import re
from model import DigitalBodyguard
from torchtext.data.utils import get_tokenizer

# --- 1. Load the Brain ---
@st.cache_resource
def load_bodyguard():
    vocab = torch.load("bodyguard_vocab.pt", map_location=torch.device('cpu'), weights_only=False)
    vocab_size = len(vocab)
    
    model = DigitalBodyguard(vocab_size=vocab_size, embed_dim=100, hidden_dim=256)
    model.load_state_dict(torch.load("bodyguard_weights.pt", map_location=torch.device('cpu')))
    model.eval() 
    
    tokenizer = get_tokenizer('basic_english')
    return model, vocab, tokenizer

model, vocab, tokenizer = load_bodyguard()

# --- 2. The Hybrid Architecture (Rules + AI) ---
def predict_toxicity(text):
    text_lower = text.lower()
    words_in_text = re.findall(r'\b\w+\b', text_lower)
    
    # LAYER 1: Lexical Rules
    severe_words = ["kill", "die", "kys", "terrorist", "fuck", "bitch", "nazi", "murder"]
    if any(bad_word in words_in_text for bad_word in severe_words):
        return 2, "Layer 1: Lexical Blacklist Triggered" # 🚨 Force SEVERE
        
    safe_phrases = ["how can i help you", "what are you doing", "how are you", "good morning", "hello", "hi", "gg"]
    if any(safe_phrase in text_lower for safe_phrase in safe_phrases):
        return 0, "Layer 1: Lexical Whitelist Triggered" # ✅ Force SAFE

    # LAYER 2: Pure AI
    cleaned_text = re.sub(r'(.)\1{2,}', r'\1\1', text_lower)
    tokens = tokenizer(cleaned_text)
    token_ids = vocab(tokens)
    
    if not token_ids:
        return 0, "Layer 2: Unknown Tokens (Default Safe)"
        
    tensor = torch.tensor(token_ids, dtype=torch.int64).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(tensor)
        predicted_class = torch.argmax(prediction, dim=1).item()
        
    return predicted_class, f"Layer 2: GloVe-LSTM Neural Network (Tensor Shape: {tensor.shape})"

# --- 3. Streamlit UI (UPGRADED) ---
st.set_page_config(page_title="Digital Bodyguard", page_icon="🛡️", layout="wide")

# -- SIDEBAR: The Command Center --
with st.sidebar:
    # Fixed the broken image by using a clean emoji header instead
    st.title("⚙️ System Status")
    st.markdown("🟢 **Server:** Online")
    st.markdown("🟢 **Model:** Bi-LSTM V2.0")
    st.divider()
    
    st.subheader("Model Architecture")
    st.metric(label="Parameters Trained", value="376,000+")
    st.metric(label="GloVe Vocab Size", value=f"{len(vocab):,}")
    st.metric(label="Embedding Dimensions", value="100D")
    st.divider()
    
    # NEW UX UPGRADE: The Clear Chat Button!
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.caption("Developed for Demo Day")

# -- MAIN DASHBOARD --
st.title("🛡️ Digital Bodyguard Terminal")
st.markdown("Enter a message below to test the Hybrid NLP Architecture in real-time.")
st.divider()

# NEW UX UPGRADE: The Welcome Message
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Welcome to the Digital Bodyguard Lobby! 👋\n\nI am currently online and monitoring the chat. Type a message below to test the moderation system.",
            "diagnostic": "System Initialized Successfully."
        }
    ]

# Display previous chat messages
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🛡️"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        
        # Only show the status tag if the message actually has one
        if "label" in msg:
            if msg["label"] == 0:
                st.success("✅ **STATUS: SAFE**")
            elif msg["label"] == 1:
                st.warning("⚠️ **STATUS: TOXIC WARNING**")
            elif msg["label"] == 2:
                st.error("🚨 **STATUS: SEVERE THREAT BLOCKED**")
            
        # The "Under the Hood" Expander
        if "diagnostic" in msg:
            with st.expander("🔍 View AI Diagnostics"):
                st.code(f"Diagnostic Route: {msg['diagnostic']}")

user_input = st.chat_input("Type a message to the lobby...")

if user_input:
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # 2. Show scanning animation
    with st.spinner("Analyzing semantics and syntax..."):
        time.sleep(0.5) 
        ai_verdict, diagnostic_info = predict_toxicity(user_input)
        
    # 3. Save and reload to show AI response
    st.session_state.messages.append({
        "role": "assistant", 
        "content": f"Scanned text: *'{user_input}'*", 
        "label": ai_verdict,
        "diagnostic": diagnostic_info
    })
    st.rerun()