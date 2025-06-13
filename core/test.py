import asyncio
import sys
import os
import time
from dotenv import load_dotenv

# Add the parent directory to the path so we can import from core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.tools import Tools
from core.llm import UltimateReVALAgent

load_dotenv(".env.local")

async def main():
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a timestamped log file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(logs_dir, f"agent_debug_{timestamp}.log")
    
    print(f"Debug logs will be saved to: {log_file}")
    
    KEY = os.getenv("OPENROUTER_API_KEY")
    if not KEY:
        print("Warning: OPENROUTER_API_KEY not found in .env.local")
        
    # Set HTTP_REFERER and X_TITLE for OpenRouter
    os.environ["HTTP_REFERER"] = "https://synapsereval.local"
    os.environ["X_TITLE"] = "SynapseREval"

    # Create tool instances
    tools_instance = Tools()
    
    agent = UltimateReVALAgent(
        model="deepseek/deepseek-chat-v3-0324:free",
        tool_support=True,
        temperature=0.2,
        max_model_tokens=16_000,
        max_response_tokens=2_048,
        persona_prompt=(
            "You are ReVAL, an elite autonomous agent known for rigorous reasoning "
            "and brutal honesty."
        ),
        debug=True,
        debug_log_file=log_file,
    )

    # Register tools from instances
    agent.register_tools_from_instance(tools_instance)
    
    print("Asking agent: Electrical consumption Dashboard ?")
    response = await agent.chat("Total Electrical consumption from database that the tool is connected to , to discover the db schema use tools available. Give me the answer in a dashboard using the tools available.The dashboard should contain multiple charts ( that are availble in the tools) and a table, periodically ")
    
    print(f"\nFull debug logs are available in: {log_file}")

if __name__ == "__main__":
    asyncio.run(main())
