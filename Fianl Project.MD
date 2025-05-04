# AI Agents for Robotics - Final Project

## Project Overview
In this project we use AI agents based on the ReAct principles for robotics. Our aim is to design a system whose principles are based on earlier work at JPL that allows the development of agents capable of supporting real time interactions with robots for algorithmic design (simulation mode) or field verification/testing and monitoring. Due to the time constraints of this project the next paragraph specializes the project goals into a narrower scope, with the understanding that what is built can be considered to be a core capability that can be extended in the future.

## 1. ROSA Milestone Replication

**Youtube Video:** [ROSA Replication Demo](https://www.youtube.com/watch?v=FVrqXn0KEf4)

### Environment Setup
- Used the provided maze environment with Gazebo simulator and RViz
- Implemented robot control and visualization

<p align="center">
  <img src="https://github.com/Fitz-Yuan/AI-for-Robotics/raw/main/project3/Screenshots/Task2_3.png?raw=true" alt="1" width="500">
</p>

### LLM API Integration
- Utilized LLM API for reasoning capabilities
- Resource: [Free LLM API Resources](https://github.com/cheahjs/free-llm-api-resources)
- Implementation used Llama3.1 through together.ai

### ROSA Chat Replication
Successfully recreated the conversational interactions from the ROSA paper:

<p align="center">
  <img src="https://github.com/Fitz-Yuan/AI-for-Robotics/blob/main/Fianl_projects/screenshots/1.png?raw=true" alt="2" width="500">
</p>

<p align="center">
  <img src="https://github.com/Fitz-Yuan/AI-for-Robotics/blob/main/Fianl_projects/screenshots/2.png?raw=true" alt="2" width="500">
</p>

<p align="center">
  <img src="https://github.com/Fitz-Yuan/AI-for-Robotics/blob/main/Fianl_projects/screenshots/3.png?raw=true" alt="2" width="500">
</p>

<p align="center">
  <img src="https://github.com/Fitz-Yuan/AI-for-Robotics/blob/main/Fianl_projects/screenshots/4.png?raw=true" alt="2" width="500">
</p>


## 2. ROSA Enhancement Milestone

**Youtube Video:** [ROSA Enhancement Demo](https://www.youtube.com/watch?v=GfolwEMVQ-s)

### Langchain to pydantic_ai Transition
Replaced the original Langchain framework with pydantic_ai for improved agent functionality definition:

#### Agent Definition
```python
from pydantic_ai import Agent

# API settings
os.environ["OPENAI_API_KEY"] = "api_key"
os.environ["OPENAI_API_BASE"] = "https://api.together.xyz/v1"

# Create the agent
agent = Agent(
    name="ROSA",
    description="""
    ROSA
    """,
    llm="together"
)
```

#### Command Handling Implementation
```python
def handle_command(command: str):
    if re.search(r'(?:go|move)\s+forward', command, re.IGNORECASE):
        distance_match = re.search(r'(\d+\.?\d*)\s*(?:meter|m)', command.lower())
        distance = float(distance_match.group(1)) if distance_match else 1.0
        return move_robot(distance)
    
    elif re.search(r'go to (the|a)?\s+(\w+)', command, re.IGNORECASE):
        # Extract semantic target
        match = re.search(r'go to (the|a)?\s+(\w+)', command, re.IGNORECASE)
        target = match.group(2) if match else None
        return navigate_to_object(target)
```

### Semantic Navigation Implementation
Added advanced navigation capabilities:
- **Semantic Map:** Stores object locations and descriptions
- **Navigation Planner:** Plans paths to semantic targets
- **LLM Reasoning:** Uses LLM to interpret natural language navigation commands

Implementation files are located in `Fianl_projects/rosa_agent_controller` folder.

### Example Results
<p align="center">
  <img src="https://github.com/Fitz-Yuan/AI-for-Robotics/blob/main/Fianl_projects/screenshots/Screenshot%202025-05-04%20at%205.29.33%20PM.png?raw=true" alt="2" width="500">
</p>

## Conclusion
By replacing Langchain with pydantic_ai and implementing semantic navigation, I created a more efficient and capable robot control system. The agent can now understand complex navigation commands and successfully navigate to semantic targets in the environment.

Please note that as I needed to record videos from two screens simultaneously, the text in some parts of the recordings may appear small and difficult to read. I apologize for any inconvenience this might cause when reviewing the demonstration.
