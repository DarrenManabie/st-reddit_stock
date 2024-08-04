import streamlit as st
import praw
import prawcore
import datetime
import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter

# Load environment variables from .env file if it exists
if os.path.exists(".env"):
    load_dotenv()

# Function to get environment variables, with fallback to Streamlit secrets
def get_env_variable(var_name):
    return os.getenv(var_name) or st.secrets.get(var_name)

# Set up Reddit API client
try:
    reddit = praw.Reddit(
        client_id=get_env_variable("REDDIT_CLIENT_ID"),
        client_secret=get_env_variable("REDDIT_CLIENT_SECRET"),
        user_agent=get_env_variable("REDDIT_USER_AGENT")
    )
    # Verify the credentials by making a simple API call
    reddit.user.me()
except prawcore.exceptions.ResponseException as e:
    st.error(f"Error authenticating with Reddit API: {str(e)}")
    st.error("Please check your Reddit API credentials in the .env file or Streamlit secrets.")
    st.stop()
except prawcore.exceptions.OAuthException as e:
    st.error(f"OAuth Error: {str(e)}")
    st.error("Please verify your Reddit API credentials and ensure they have the correct permissions.")
    st.stop()

# Set up language model
OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"  # Changed to a more commonly available model

model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=MODEL)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Rest of the code remains the same...

# (Include all the functions: get_reddit_posts, setup_chain, analyze_posts, and main)

if __name__ == "__main__":
    main()
