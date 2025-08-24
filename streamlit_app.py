import requests
import streamlit as st

# Replace with your FastAPI or any other endpoint
API_URL = " http://langgraph-app-lb-271526777.us-east-1.elb.amazonaws.com/answer_user_query"

st.set_page_config(page_title="WixQA Chatbot", layout="centered")

st.title("WixQA Chatbot")

# Input box
query = st.text_input("Enter your query:")

if st.button("Submit") and query.strip():
    try:
        # Send request
        response = requests.post(API_URL, json={"query": query})
        response.raise_for_status()

        # Assume API returns JSON with {"answer": "...markdown..."}
        result = response.json().get("answer", "No answer found.")

        # Display markdown
        st.markdown(result)
        st.markdown(response.json().get("metrics", "No metrics found."))

    except Exception as e:
        st.error(f"Error: {e}")
