# examples/test_case.py

import asyncio
import os
import sys
import time
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.llm import UltimateReVALAgent
from core.synapses import SynapseNetwork, Neuron
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env.local")

# --- ANSI Colors for beautiful printing ---
class C:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{C.HEADER}{C.BOLD}--- {text} ---{C.END}")

def print_info(text):
    print(f"{C.CYAN}INFO: {text}{C.END}")

def print_success(text):
    print(f"{C.GREEN}SUCCESS: {text}{C.END}")

def print_warning(text):
    print(f"{C.YELLOW}WARNING: {text}{C.END}")
    
def print_weights(net: SynapseNetwork):
    print(f"{C.BLUE}Current Synapse Weights:{C.END}")
    for u, v, data in net.G.edges(data=True):
        weight = data['syn'].weight
        color = C.GREEN if weight > 1.0 else C.YELLOW if weight < 1.0 else C.BLUE
        print(f"  {u} -> {v}: {color}{weight:.4f}{C.END}")
    print("-" * 20)

async def main():
    """
    This test case demonstrates the core powers of SynapseReVAL:
    1.  Adaptive Routing via Softmax.
    2.  Synaptic Learning via STDP.
    3.  Autonomous Tool Generation by an agent.
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a timestamped log file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(logs_dir, f"full_test_case_{timestamp}.log")
    
    print(f"Full debug logs will be saved to: {log_file}")
    
    # Set HTTP_REFERER and X_TITLE for OpenRouter
    os.environ["HTTP_REFERER"] = "https://synapsereval.local"
    os.environ["X_TITLE"] = "SynapseREval"

    # Agent factory function to create specialized agents
    def agent_factory(name: str, role: str, persona_prompt: str):
        return UltimateReVALAgent(
            model="deepseek/deepseek-chat-v3-0324:free",
            tool_support=True,
            temperature=0.1,  # Low temp for predictable behavior
            max_model_tokens=16_000,
            max_response_tokens=2_048,
            persona_prompt=persona_prompt,
            debug=True,
            debug_log_file=log_file,
        )

    # --- SCENARIO SETUP ---
    print_header("SCENARIO: The Network That Learned to Calculate Square Roots")
    print_info("A router agent must decide which specialist can answer a math question.")
    print_info("One specialist is a 'Philosopher' (bad choice), the other is a 'Coder' (good choice).")
    print_info("Initially, the Coder doesn't know how to calculate square roots and must build a tool for it.")
    
    # --- NETWORK INITIALIZATION ---
    net = SynapseNetwork(learning_rate=0.1) # Higher learning rate to see changes faster

    # Create persona prompts
    router_prompt = (
        "You are a Router. Your SOLE function is to choose the best specialist for a task. "
        "You MUST NOT answer the user's question yourself. "
        "First, think step-by-step about the user's request. "
        "Second, in ONE sentence, state which specialist ('Coder' or 'Philosopher') you choose and why. "
        "Your output MUST be ONLY that one sentence."
    )
    
    philosopher_prompt = (
        "You are a Philosopher. You ONLY discuss the abstract, non-mathematical nature of concepts. "
        "You MUST NOT perform calculations or give concrete answers. "
        "If asked about 'the square root of 256', you should discuss the concept of 'rootness' or 'sixteen-ness', not the number 16."
    )
    
    coder_prompt = (
        "You are a Coder. You ONLY solve problems by using tools. You are incapable of answering mathematical questions directly. "
        "Your process MUST be: "
        "1. Analyze the request: 'What is the square root of 256?'. "
        "2. Check available tools. Do I have a 'sqrt' tool? "
        "3. If a 'sqrt' tool exists, you MUST call it with the correct arguments. "
        "4. If a 'sqrt' tool does NOT exist, you MUST call `create_and_test_tool` to build it. Provide the necessary python_code and test_code. "
        "You MUST call a tool. Do NOT provide a final answer in text."
    )
    
    finalizer_prompt = (
        "You are a Finalizer. Your job is to take the final, correct answer from a specialist and present it clearly to the user. "
        "Your output should be a single sentence, like: 'The final answer is X.'"
    )

    # Add neurons
    net.add_neuron("Router", Neuron(agent_factory("Router", "router", router_prompt), threshold=1.0))
    net.add_neuron("Philosopher", Neuron(agent_factory("Philosopher", "philosopher", philosopher_prompt), threshold=1.0))
    net.add_neuron("Coder", Neuron(agent_factory("Coder", "coder", coder_prompt), threshold=1.0))
    net.add_neuron("Finalizer", Neuron(agent_factory("Finalizer", "finalizer", finalizer_prompt), threshold=1.0))

    # Connect neurons
    print_header("INITIAL NETWORK TOPOLOGY")
    net.connect("Router", "Philosopher", weight=1.0) # Initially, equal weights
    net.connect("Router", "Coder", weight=1.0)
    net.connect("Philosopher", "Finalizer", weight=0.5) # Weak connection, as it's a bad path
    net.connect("Coder", "Finalizer", weight=1.5)       # Strong connection, this is the good path
    
    print_weights(net)

    # --- RUN 1: The First Attempt ---
    print_header("RUN 1: Asking the network to calculate sqrt(256)")
    initial_prompt = "What is the square root of 256?"
    
    # We will run the network multiple times to see it learn
    history = await net.run(
        max_cycles=8,
        entry_neuron="Router",
        initial_prompt=initial_prompt,
        neuron_delay=1.0
    )

    print_header("ANALYSIS OF RUN 1")
    
    # Check if the Coder created a new tool
    if 'sqrt' in net.neurons['Coder'].agent.tools:
        print_success("The 'Coder' agent successfully created a `sqrt` tool for itself!")
        # Let's inspect the created tool
        tool_spec = net.neurons['Coder'].agent.tools['sqrt']
        print_info(f"  Tool Name: {tool_spec.name}")
        print_info(f"  Description: {tool_spec.description}")
    else:
        print_warning("The Coder agent did not create the `sqrt` tool in this run.")

    # Show the learned weights
    print_header("NETWORK STATE AFTER RUN 1 (STDP Learning)")
    print_info("Because Coder -> Finalizer was a successful path, its weight should increase.")
    print_info("The Router -> Coder connection should also be strengthened.")
    print_weights(net)
    
    # Extract final answer
    final_answer = next((entry["payload"] for entry in reversed(history) if entry["neuron"] == "Finalizer"), "No final answer produced.")
    print_header("FINAL ANSWER from Run 1")
    print(f"{C.GREEN}{final_answer}{C.END}")
    

    # --- RUN 2: Reinforcing the Learned Path ---
    print_header("RUN 2: Asking a similar question: sqrt(144)")
    print_info("The network should now favor the 'Coder' path due to learned weights.")
    print_info("The Coder should use its newly created `sqrt` tool directly.")
    
    # Clear history and inboxes for a fresh run with the learned weights
    net.history.clear()
    for n in net.neurons.values():
        n.inbox.clear()
        
    second_prompt = "Quickly, what's the square root of 144?"
    
    history_2 = await net.run(
        max_cycles=5,
        entry_neuron="Router",
        initial_prompt=second_prompt,
        neuron_delay=1.0
    )
    
    print_header("ANALYSIS OF RUN 2")
    final_answer_2 = next((entry["payload"] for entry in reversed(history_2) if entry["neuron"] == "Finalizer"), "No final answer produced.")
    print_header("FINAL ANSWER from Run 2")
    print(f"{C.GREEN}{final_answer_2}{C.END}")

    print_header("FINAL NETWORK STATE")
    print_info("The weights for the Coder -> Finalizer path should be even stronger now.")
    print_weights(net)

    # Save the learned brain state
    brain_path = Path(os.path.join(os.path.dirname(__file__), '..', 'brains', f"learned_sqrt_brain_{timestamp}.json"))
    brain_path.parent.mkdir(exist_ok=True)
    net.save(str(brain_path))
    print_success(f"Learned brain state saved to: {brain_path}")


if __name__ == "__main__":
    # Note: This test makes multiple, sequential LLM calls and may take a few minutes to run.
    asyncio.run(main())