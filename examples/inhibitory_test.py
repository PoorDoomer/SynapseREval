# examples/inhibitory_test.py

import asyncio
import os
import sys
import time
from pathlib import Path

# Set Python's output encoding to UTF-8 for Windows console compatibility
os.environ["PYTHONIOENCODING"] = "utf-8"
# Check if Docker is available by trying to run 'docker --version'
try:
    import subprocess
    result = subprocess.run(['docker', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    DOCKER_AVAILABLE = result.returncode == 0
    if DOCKER_AVAILABLE:
        print("Docker is available")
    else:
        print("Docker is not available")
except Exception as e:
    print(f"Error checking Docker availability: {e}")
    DOCKER_AVAILABLE = False
# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.llm import UltimateReVALAgent, tool
from core.synapses import SynapseNetwork, Neuron
from dotenv import load_dotenv

load_dotenv(".env.local")

# --- ANSI Colors for rich terminal output ---
class C:
    if os.name == 'nt':
        # Simple text for Windows to avoid encoding issues
        HEADER, BLUE, CYAN, GREEN, YELLOW, RED, END, BOLD, UNDERLINE = ("", "", "", "", "", "", "", "", "")
    else:
        # Full colors for other systems
        HEADER, BLUE, CYAN, GREEN, YELLOW, RED, END, BOLD, UNDERLINE = ('\033[95m', '\033[94m', '\033[96m', '\033[92m', '\033[93m', '\033[91m', '\033[0m', '\033[1m', '\033[4m')

def print_header(text): print(f"\n{C.HEADER}{C.BOLD}--- {text} ---{C.END}")
def print_info(text): print(f"{C.CYAN}INFO: {text}{C.END}")
def print_success(text): print(f"{C.GREEN}SUCCESS: {text}{C.END}")
def print_warning(text): print(f"{C.YELLOW}WARNING: {text}{C.END}")
    
def print_weights(net: SynapseNetwork):
    print(f"{C.BLUE}Current Synapse Weights:{C.END}")
    for u, v, data in net.G.edges(data=True):
        weight = data['syn'].weight
        color = C.GREEN if weight > 0 else C.RED
        print(f"  {u} -> {v}: {color}{weight:.4f}{C.END}")
    print("-" * 20)

class SemanticRouterTools:
    """Tools for the advanced SemanticRouter agent."""
    def __init__(self, network: SynapseNetwork):
        self.network = network
        self.original_prompt = ""

    @tool("Sends weighted excitatory signals to specialists based on semantic analysis.", expected_runtime=0.1)
    async def send_weighted_signals(self, coder_relevance: float, philosopher_relevance: float):
        """
        Sends signals to specialists. Relevance must be a float between 0.0 and 1.0.
        Args:
            coder_relevance: How relevant the task is to the Coder (logic, math, code).
            philosopher_relevance: How relevant the task is to the Philosopher (abstract, subjective).
        """
        # Activation is scaled to make the difference more pronounced
        coder_activation = coder_relevance * 2.0
        philosopher_activation = philosopher_relevance * 2.0
        
        self.network.inject("Coder", self.original_prompt, weight=coder_activation)
        self.network.inject("Philosopher", self.original_prompt, weight=philosopher_activation)
        
        message = f"Signals sent: Coder activation={coder_activation:.2f}, Philosopher activation={philosopher_activation:.2f}"
        print_info(message)
        return message

async def main():
    print_header("DEFINITIVE TEST: INHIBITION, TOOL-BUILDING, AND LEARNING")
    
    log_file = os.path.join('logs', f"inhibitory_test_{time.strftime('%Y%m%d-%H%M%S')}.log")
    print(f"Debug logs will be saved to: {log_file}")
    
    os.environ["HTTP_REFERER"] = "https://synapsereval.local"
    os.environ["X_TITLE"] = "SynapseREval"

    def agent_factory(name: str, persona_prompt: str, include_defaults: bool = True):
        return UltimateReVALAgent(
            model="deepseek/deepseek-chat-v3-0324:free",
            tool_support=True, temperature=0.1, debug=True, debug_log_file=log_file,
            persona_prompt=persona_prompt,
            register_default_tools=include_defaults
        )

    # --- Network and Personas ---
    net = SynapseNetwork(learning_rate=0.2) # High learning rate to see STDP changes clearly
    semantic_router_tools = SemanticRouterTools(net)
    
    router_prompt = (
        "You are a highly specialized Semantic Router. Your ONLY function is to analyze a user's prompt and call the `send_weighted_signals` tool. "
        "You are FORBIDDEN from answering the user's question or calling any other tool. "
        "Analyze the user's request, then call the `send_weighted_signals` tool with relevance scores (a float from 0.0 to 1.0) for each specialist."
    )
    coder_prompt = (
        "You are a highly specialized Coder agent. You are FORBIDDEN from answering any user question directly, especially math. Your ONLY purpose is to use tools. "
        "Your thinking process MUST be followed exactly: "
        "1. Analyze the user's request (e.g., 'calculate 512 divided by 16'). "
        "2. Identify the required operation (e.g., 'division'). "
        "3. Check your available tools. Do you have a tool for this operation? "
        "4. **If you do not have the tool:** Your ONLY response must be a call to `create_and_test_tool`. "
        "5. **If the tool creation is successful:** You MUST immediately re-evaluate the ORIGINAL user request from the chat history and call the newly created tool to solve it. "
        "6. **If you already have the tool:** Your ONLY response must be a call to that tool to solve the user's request. "
        "Do not write any text, explanation, or apology. Respond ONLY with a tool call."
    )
    philosopher_prompt = "You are a Philosopher. You ONLY discuss abstract concepts. You must never provide a concrete or numerical answer."
    finalizer_prompt = "You are a Finalizer. Your only job is to receive the final, computed result from a specialist and present it clearly in the format: 'The final answer is: [result]'."

    # --- Agent and Neuron Setup ---
    # Create the router agent with NO default tools
    router_agent = agent_factory("SemanticRouter", router_prompt, include_defaults=False)
    # Then register ONLY the one tool it needs
    router_agent.register_tool(semantic_router_tools.send_weighted_signals)
    
    # The other agents get the full default toolset
    coder_agent = agent_factory("Coder", coder_prompt, include_defaults=True)
    philosopher_agent = agent_factory("Philosopher", philosopher_prompt, include_defaults=True)
    finalizer_agent = agent_factory("Finalizer", finalizer_prompt, include_defaults=True)
    
    net.add_neuron("SemanticRouter", Neuron(router_agent, threshold=0.5, role="router"))
    net.add_neuron("Coder", Neuron(coder_agent, threshold=0.8, role="specialist"))
    net.add_neuron("Philosopher", Neuron(philosopher_agent, threshold=0.8, role="specialist"))
    net.add_neuron("Finalizer", Neuron(finalizer_agent, threshold=1.0, role="finalizer"))

    # --- Synapse Connections: The Winner-Take-All Circuit ---
    print_header("NETWORK TOPOLOGY WITH LATERAL INHIBITION")
    
    # Lateral inhibition: Specialists suppress each other
    net.connect("Coder", "Philosopher", weight=-2.5, plastic=False)
    net.connect("Philosopher", "Coder", weight=-2.5, plastic=False)
    
    # Excitatory pathways to the Finalizer, these are plastic and will learn
    net.connect("Coder", "Finalizer", weight=1.5, plastic=True)
    net.connect("Philosopher", "Finalizer", weight=1.0, plastic=True)

    print_weights(net)

    # --- RUN SCENARIO: A CLEAR MATH PROBLEM ---
    print_header("RUN 1: Clear Math Problem - 'What is 512 / 16?'")
    math_prompt = "Please calculate 512 divided by 16."
    
    # Set the prompt for the router's tool
    semantic_router_tools.original_prompt = math_prompt

    history = await net.run(
        max_cycles=10, # Give it enough cycles for the multi-step tool process
        entry_neuron="SemanticRouter",
        initial_prompt=math_prompt,
        neuron_delay=0.5
    )
    
    print_header("ANALYSIS OF RUN 1")
    print_info("Expected outcome:")
    print_info("  1. Router sends high activation to Coder, low to Philosopher.")
    print_info("  2. Coder's activation overcomes its threshold and the Philosopher's weak signal.")
    print_info("  3. Coder fires, inhibiting the Philosopher and starting its task.")
    print_info("  4. Coder realizes it lacks a 'divide' tool and calls `create_and_test_tool`.")
    print_info("  5. Coder uses the new tool, gets the result, and passes it to the Finalizer.")
    print_info("  6. The `Coder -> Finalizer` synapse is strengthened by STDP.")
    
    if 'divide' in net.neurons['Coder'].agent.tools:
        print_success("The 'Coder' agent successfully built and registered a 'divide' tool!")
    else:
        print_warning("The Coder agent FAILED to build the 'divide' tool.")
    
    print_weights(net)

    # ------------------- NEW SECTION START -------------------
    # Extract and print the final answer from the network's history
    print_header("FINAL RESULT OF RUN 1")
    final_answer = "No final answer produced by Finalizer."
    for entry in reversed(history):
        if entry.get("neuron") == "Finalizer":
            final_answer = entry.get("payload", "Finalizer fired but had no payload.")
            break
            
    print(f"{C.GREEN}{final_answer}{C.END}")
    # -------------------- NEW SECTION END --------------------


if __name__ == "__main__":
    # Ensure Docker is running before executing this test case.
    if not DOCKER_AVAILABLE:
        print(f"{C.RED}{C.BOLD}WARNING: Docker is not detected. The `create_and_test_tool` will fail.{C.END}")
        print(f"{C.YELLOW}Please install Docker and ensure the Docker daemon is running to see the full test.{C.END}")
    
    asyncio.run(main())