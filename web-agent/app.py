import streamlit as st
import os
from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.groq import Groq

groq_api_key = st.text_input("Enter your Groq API key:", type="password")

if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key

if groq_api_key:
    web_agent = Agent(
        name="Web Agent",
        model=Groq(id="llama-3.1-8b-instant"),
        tools=[DuckDuckGo()],
        instructions=["Always include sources"],
        show_tool_calls=True,
        markdown=True,
    )

user_query = st.text_input("Enter your query:")
if user_query and groq_api_key:
    with st.spinner("Fetching response..."):
        response = web_agent.run(user_query)
    st.write(response)
else:
    st.warning("Please enter your Groq API key to use the agent.")
