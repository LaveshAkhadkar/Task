import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from expansion import generate_expanded_query
from classification import predict_with_loaded_model

load_dotenv()

st.title("Chatbot with BERT Classification and BART Expansion")
st.write("Classification and expansion start after 2nd message by the user")
if "history" not in st.session_state:
    st.session_state.history = []

if "ghistory" not in st.session_state:
    st.session_state.ghistory = []

user_query = st.text_input("You:", key="input")

genai.configure(api_key=os.getenv("gemini"))

model = genai.GenerativeModel("gemini-1.5-flash", system_instruction="Always answer in short 2-3 sentences.")
chat = model.start_chat(history=st.session_state.ghistory)

#if user clicks send or presses enter with user query in input
if st.button("Send") or user_query:
    if user_query:

        if len(st.session_state.ghistory) > 1:
            last_20_messages = st.session_state.ghistory[-20:]
            flattened_chat_history = " ".join([entry["parts"] for entry in last_20_messages])
            summary = model.generate_content("Summarize this: " + flattened_chat_history)

        else:
            summary = "Not available"

        
        response = chat.send_message(user_query).text
        if len(st.session_state.ghistory) > 1:
            expanded_query = generate_expanded_query(user_query, summary)

            cat1_pred, cat2_pred = predict_with_loaded_model(expanded_query)
            topic1 = ['Entertainment','Technology','Politics','Health','Sports','General']
            topic2 = ['Movies', 'Software Development', 'Global', 'Music', 'Nutrition', 
                'TV Shows', 'Fitness', 'Mental Health', 'Cybersecurity', 'UK', 
                'Diseases', 'Blockchain', 'USA', 'AI', 'Cricket', 'Celebrities', 
                'Football', 'Basketball', 'India', 'Tennis', 'General']
            classification = f"{topic1[cat1_pred]}-{topic2[cat2_pred]}"
        else:
            expanded_query = ""
            classification = ""


        # Append user and bot responses to history
        st.session_state.history.append(
            {
                "speaker": "User",
                "text": user_query,
                "classification": classification,
                "expanded_query": expanded_query,
            }
        )
        st.session_state.history.append({"speaker": "Bot", "text": response})

        st.session_state.ghistory.append(
            {"role": "user", "parts": f"{user_query}"},
        )
        st.session_state.ghistory.append({"role": "model", "parts": f"{response}"})


        # Display the entire conversation
        for entry in st.session_state.history:
            if entry["speaker"] == "User":
                st.write(
                    f"**You:** {entry['text']} (Class: {entry['classification']}, Expanded: {entry['expanded_query']})"
                )
            else:
                st.write(f"**Bot:** {entry['text']}")

if st.button("Clear Conversation"):
    st.session_state.history = []
    st.session_state.ghistory = []

    st.write("Conversation history cleared!")