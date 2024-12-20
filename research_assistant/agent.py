from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("CEREBRAS_API_KEY")
if not api_key:
    raise ValueError("CEREBRAS_API_KEY environment variable is not set.")

cerebras_llm = LLM(
    model="cerebras/llama3.1-70b",
    api_key=api_key,
    base_url="https://api.cerebras.ai/v1",
    temperature=0.5,
)

# research agent
researcher = Agent(
    role='{topic} Senior Researcher',
    goal='Uncover groundbreaking technologies in {topic} for the year 2024',
    backstory='Driven by curiosity, you explore and share the latest innovations.',
    tools=[SerperDevTool()],
    llm=cerebras_llm
)

# define research agent's task
research_task = Task(
    description='Identify the next big trend in {topic} with pros and cons.',
    expected_output='A 3-paragraph report on emerging {topic} technologies.',
    agent=researcher
)

def main():
    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True
    )
    result = crew.kickoff(inputs={'topic': 'The future of electric vehicles'})
    print(result)

if __name__ == "__main__":
    main()