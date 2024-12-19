import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ.get("CEREBRAS_API_KEY")
if not api_key:
    st.error("CEREBRAS_API_KEY environment variable is not set. Please check your .env file.")
    st.stop()

cerebras_llm = LLM(
    model="cerebras/llama3.1-70b",
    api_key=api_key,
    base_url="https://api.cerebras.ai/v1",
    temperature=0.5,
)

researcher = Agent(
    role='{topic} Senior Researcher',
    goal='Uncover groundbreaking technologies in {topic} for the year 2024',
    backstory='Driven by curiosity, you explore and share the latest innovations.',
    tools=[SerperDevTool()],
    llm=cerebras_llm
)

def is_valid_topic(topic):
    valid_keywords = ["AI", "Artificial Intelligence", "Electric Vehicles", "Healthcare", "Biotechnology", 
                      "Robotics", "Quantum Computing", "Blockchain", "Cybersecurity", "Data Science"]
    for keyword in valid_keywords:
        if keyword.lower() in topic.lower():
            return True
    return False

def get_research_output(topic):
    research_task = Task(
        description=f'Identify the next big trend in {topic} with pros and cons.',
        expected_output=f'A 3-paragraph report on emerging {topic} technologies.',
        agent=researcher
    )

    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True
    )
    result = crew.kickoff(inputs={'topic': topic})
    return result

st.title("AI-Powered Research Assistant")
st.write("Enter a **technology** or **field** (e.g., AI, Electric Vehicles, Healthcare) to uncover groundbreaking trends.")

topic = st.text_input("Enter a technology or field:")

if st.button("Get Research Report"):
    if not topic:
        st.warning("Please enter a topic to continue.")
    elif not is_valid_topic(topic):
        st.warning("Invalid input. Please enter a valid technology or field (e.g., AI, Electric Vehicles).")
    else:
        with st.spinner("Researching..."):
            try:
                output = get_research_output(topic)
                st.success("Research completed!")
                st.subheader("Research Report")
                st.write(output)
            except Exception as e:
                st.error(f"An error occurred: {e}")
