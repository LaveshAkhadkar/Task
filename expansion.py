import json
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.environ["gemini"])


def save_history(history):
    with open("history.json", "w") as f:
        json.dump(history, f, indent=4)


def clear_history():
    with open("history.json", "w") as f:
        json.dump([], f, indent=4)


def load_history():
    try:
        with open("history.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def add_to_history(history, role, message):
    history.append({"role": role, "parts": message})
    if len(history) > 20:
        history = history[-20:]
    save_history(history)
    return history


def generate_expanded_query(user_query):

    history = load_history()
    history = history[-20:]

    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        system_instruction="""
        Expand the user query using relevant context from the conversation history.
        
        Return only the expanded query in JSON format:
        {"expanded_query": "Expanded query text here"}
        """,
    )

    chat = model.start_chat(history=history)
    response = chat.send_message(user_query)
    expanded_query = response.text.strip()

    history = add_to_history(history, "user", user_query)
    history = add_to_history(history, "model", expanded_query)
    clean_response = (
        response.text.strip().replace("```json", "").replace("```", "").strip()
    )

    # load in json format
    expanded_query = json.loads(clean_response)
    expanded_query = expanded_query["expanded_query"]
    return expanded_query


def generate_expanded_query_history(user_query, history=[]):
    history = history[-20:]

    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        system_instruction="""
        Expand the user query using relevant context from the conversation history.
        
        Return only the expanded query in JSON format:
        {"expanded_query": "Expanded query text here"}
        """,
    )

    chat = model.start_chat(history=history)
    response = chat.send_message(user_query)

    clean_response = (
        response.text.strip().replace("```json", "").replace("```", "").strip()
    )

    expanded_query = json.loads(clean_response)
    expanded_query = expanded_query["expanded_query"]
    return expanded_query
