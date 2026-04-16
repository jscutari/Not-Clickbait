import streamlit as st
import pickle
import re
import pandas as pd
from xgboost import XGBClassifier

# Load model and vectorizer
@st.cache_resource #Loads the model and vectorizer once and caches them for future use
def load_model():
    with open('models/vectorizer.pkl', 'rb') as f: #read binary mode
        vectorizer = pickle.load(f) 
    model = XGBClassifier()
    model.load_model('models/xgboost_clickbait.json')
    return vectorizer, model


def clean_text(text: str):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

if "history" not in st.session_state:
    st.session_state.history = []
    
page_bg_css = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #0e1117, #333333);
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)

try:
    st.image("/Users/jscutari/Documents/UNC/TechX/Project/Not-Clickbait/images/techx.avif", width=150) 
except FileNotFoundError:
    st.warning("Image not found")

st.title("Not Clickbait: Clickbait Detector")
st.write("Type or paste a headline to check if it's clickbait")



title = st.text_input("Enter a headline:")


if st.button("Check"):
    if title:
        vectorizer, model = load_model()
        cleaned = clean_text(title)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]

        if prediction == 1:
            st.error(f"CLICKBAIT — {probability[1]:.0%} confidence")
        else:
            st.success(f"NOT CLICKBAIT — {probability[0]:.0%} confidence")

        
        st.write("Heat Map of Word Importance:")
        
        feature_names = vectorizer.get_feature_names_out()
        importances = model.feature_importances_
        nonzero_indices = vectorized[0].nonzero()[1]
        
        word_weights = []
        
        for idx in nonzero_indices:
            word = feature_names[idx]
            weight = importances[idx]
            word_weights.append({"Word": word, "Importance Weight": weight})
            
        
        if word_weights:
            weight_df = pd.DataFrame(word_weights).sort_values(by="Importance Weight", ascending=False)
            styled_df = weight_df.style.background_gradient(cmap='Reds', subset=['Importance Weight'])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("No words were in the model vocabulary.")
            
       

        st.session_state.history.append({
            "Headline": title,
            "Result": "Clickbait" if prediction == 1 else "Not Clickbait",
            "Confidence": f"{probability[1]:.0%}" if prediction == 1 else f"{probability[0]:.0%}"
        })
    else:
        st.warning("Please enter a headline first")


if st.session_state.history:
    st.divider()
    st.subheader("Session Analytics")


    history_df = pd.DataFrame(st.session_state.history)

    
    total_checked = len(history_df)
    clickbait_count = len(history_df[history_df["Result"] == "Clickbait"])
    not_clickbait_count = total_checked - clickbait_count

    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Checked", total_checked)
    col2.metric("Clickbait", clickbait_count)
    col3.metric("Not Clickbait", not_clickbait_count)

    
    st.write("**Clickbait vs. Authentic Headlines**")
    result_counts = history_df["Result"].value_counts()
    st.bar_chart(result_counts, color="#61c0d7") 

    # 
    st.subheader("History Log")
    st.dataframe(history_df, use_container_width=True)