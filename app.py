import streamlit as st
import praw
import prawcore
import datetime
import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
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
        client_id=st.secrets["REDDIT_CLIENT_ID"],
        client_secret=st.secrets["REDDIT_CLIENT_SECRET"],
        user_agent=st.secrets["REDDIT_USER_AGENT"]
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
OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
MODEL = "gpt-3.5-turbo"  # Changed to a more commonly available model

model = ChatOpenAI(openai_api_key=OPENAI_KEY, model=MODEL)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)

# Set up prompt template
template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)

# Set up parser
parser = StrOutputParser()

def get_reddit_posts():
    subreddit = reddit.subreddit("stocks")
    subPopular = ""
    subToday = ""
    today = datetime.datetime.now().date()

    try:
        for submission in subreddit.new(limit=50):
            if submission.score > 100:
                subPopular += f"Title: {submission.title}\n"

            if datetime.datetime.fromtimestamp(submission.created_utc).date() == today and submission.score > 20:
                subToday += f"Title: {submission.title}\n"
    except prawcore.exceptions.ResponseException as e:
        st.error(f"Error fetching posts from Reddit: {str(e)}")
        return None, None

    return subPopular, subToday

def setup_chain(docs):
    vectorstore = DocArrayInMemorySearch.from_documents(docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | prompt
        | model
        | parser
    )
    return chain

def analyze_posts(subPopular, subToday):
    if subPopular is None or subToday is None:
        return None

    docs = [
        Document(page_content=subPopular, metadata={"source": "local"}),
        Document(page_content=subToday, metadata={"source": "local"})
    ]
    chain = setup_chain(docs)

    questions = [
        "You are a seasoned stock analyst with expertise in quickly skimming through daily news to identify major events impacting the stock market. You will be provided with a list of news titles related to the stock market. Your task is to identify the most significant news title based on its potential impact on the market. Please consider factors such as market trends, company importance, and potential financial implications. What is the biggest news title today?",
        "You are an executive assistant to a busy enterprise CEO who values every minute of his time. Your task is to provide a concise and relevant overview of today's stock market news. Focus on major market movements, significant company news, and key economic events that could impact the CEO's decisions. Extract key news headlines into structured format. Generate a concise summary from the structured details",
        "You are given a list of stock market news titles. Your task is to organize and display these titles by company or topic. If multiple titles relate to the same company or topic, combine and summarize them into a single entry. Ensure that the responses are numbered and any duplicate titles are removed. Present the final organized list clearly and concisely.",
        "You are given a list of stock market news titles. What is the overall sentiment. Give your answer as positive or negative or neutral only.",
    ]

    results = []
    for question in questions:
        results.append(chain.invoke({"question": question}))

    return results

def main():
    st.title("Reddit Stocks Analysis")

    if st.button("Fetch and Analyze Posts"):
        with st.spinner("Fetching posts from Reddit..."):
            subPopular, subToday = get_reddit_posts()

        if subPopular is not None and subToday is not None:
            with st.spinner("Analyzing posts..."):
                results = analyze_posts(subPopular, subToday)

            if results:
                st.subheader("Analysis Results")
                st.write("Most Significant News:")
                st.write(results[0])

                st.write("Executive Summary:")
                st.write(results[1])

                st.write("Organized News by Company/Topic:")
                st.write(results[2])

                st.write("Overall Sentiment:")
                st.write(results[3])
            else:
                st.error("An error occurred during analysis. Please check the console for more information.")
        else:
            st.error("Failed to fetch posts from Reddit. Please check your API credentials and try again.")

if __name__ == "__main__":
    main()
