# SynapseREval

A powerful autonomous agent implementing the ReVAL (Reason‑Verify‑Adapt‑Loop) architecture with confidence gating and self-reflection.

## Features

- **ReVAL loop** with confidence gating & self-reflection
- **Tool-calling** strategy: native OpenAI function-calling or JSON fallback
- **ScratchPad** with TTL + automatic large-payload off-loading
- Built-in **meta-tools** (goal-state store, complexity estimator, verifier, etc.)
- Automatic **toolsmith** (create & test new tools on-the-fly)
- **OpenRouter support** for using various LLM providers

## Setup

1. Clone the repository
2. Create a `.env.local` file with your API keys:
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   HTTP_REFERER=your_site_url_here
   X_TITLE=your_site_name_here
   ```
3. Install the required dependencies:
   ```
   pip install tiktoken pydantic python-dotenv openai>=1.0.0
   ```

## Usage

Basic usage:

```python
import asyncio
from core.llm import UltimateReVALAgent
from core.tools import Tools

async def main():
    # Create tools instance
    tools = Tools()
    
    # Initialize agent
    agent = UltimateReVALAgent(
        model="deepseek/deepseek-chat-v3-0324:free",  # OpenRouter model
        tool_support=True,
        temperature=0.2,
        debug=True  # Enable debug mode
    )
    
    # Register tools
    agent.register_tools_from_instance(tools)
    
    # Chat with the agent
    response = await agent.chat("What is the sum of 1 and 2?")
    print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Debug Mode

The agent includes a comprehensive debug mode that logs detailed information about the agent's execution. This is useful for understanding what's happening at each step of the ReVAL loop and debugging issues.

### Enabling Debug Mode

```python
agent = UltimateReVALAgent(
    model="deepseek/deepseek-chat-v3-0324:free",
    debug=True,  # Enable debug mode
    debug_log_file="logs/agent_debug.log"  # Optional: Save logs to a file
)
```

### Debug Output

The debug output includes:

- **Step-by-step execution**: Each step in the agent's execution is logged with a timestamp
- **LLM requests and responses**: Detailed information about the requests sent to the LLM and the responses received
- **Tool calls and results**: Information about tool calls and their results
- **Conversation state**: The current state of the conversation at key points

### Example Debug Output

```
2025-06-13 19:30:14,403 - UltimateReVAL - DEBUG - Initializing UltimateReVALAgent with model: deepseek/deepseek-chat-v3-0324:free
2025-06-13 19:30:14,403 - UltimateReVAL - DEBUG - OpenRouter mode: True
2025-06-13 19:30:14,403 - UltimateReVAL - DEBUG - STEP: Registering built-in tools
...
2025-06-13 19:30:17,386 - UltimateReVAL - DEBUG - STEP: Extracted 1 tool calls
2025-06-13 19:30:17,387 - UltimateReVAL - DEBUG - STEP: Executing tool call: sum
2025-06-13 19:30:17,388 - UltimateReVAL - DEBUG - STEP: Executing tool: sum with args: {'a': 1, 'b': 2}
...
2025-06-13 19:30:19,387 - UltimateReVAL - DEBUG - STEP: No tool calls detected, returning final response
```

### Debugging Tips

1. **Enable file logging**: Use the `debug_log_file` parameter to save logs to a file for later analysis
2. **Check conversation state**: Look at the conversation state dumps to understand what the agent is "thinking"
3. **Monitor tool calls**: Pay attention to tool calls and their results to debug issues with tools
4. **Examine LLM requests**: Look at the LLM requests and responses to understand what's being sent to the model

## Creating Custom Tools

You can create custom tools by using the `@tool` decorator:

```python
from core.llm import tool

class MyTools:
    @tool("Add two numbers together")
    def add(self, a: int, b: int) -> int:
        return a + b
        
    @tool("Multiply two numbers")
    def multiply(self, a: int, b: int) -> int:
        return a * b
```

Then register your tools with the agent:

```python
my_tools = MyTools()
agent.register_tools_from_instance(my_tools)
``` 