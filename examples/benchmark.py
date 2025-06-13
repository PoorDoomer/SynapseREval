# examples/benchmark.py

import asyncio
import os
import sys
import time
import csv
from pathlib import Path
from dataclasses import dataclass, field

from inhibitory_test import SemanticRouterTools

# Set Python's output encoding to UTF-8 for Windows console compatibility
os.environ["PYTHONIOENCODING"] = "utf-8"

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.llm import UltimateReVALAgent, tool
from core.synapses import SynapseNetwork, Neuron
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv(".env.local")

# --- ANSI Colors / Rich Console Setup ---
console = Console()
C_HEADER = "bold magenta"
C_SUCCESS = "bold green"
C_FAILURE = "bold red"
C_INFO = "cyan"
C_NEUTRAL = "blue"
C_WEIGHT_GOOD = "green"
C_WEIGHT_BAD = "red"

def print_header(text): console.print(f"\n--- {text} ---", style=C_HEADER)
def print_info(text): console.print(f"INFO: {text}", style=C_INFO)
def print_success(text): console.print(f"SUCCESS: {text}", style=C_SUCCESS)
def print_failure(text): console.print(f"FAILURE: {text}", style=C_FAILURE)

def print_weights(net: SynapseNetwork):
    console.print("Current Synapse Weights:", style=C_NEUTRAL)
    for u, v, data in net.G.edges(data=True):
        weight = data['syn'].weight
        style = C_WEIGHT_GOOD if weight > 0 else C_WEIGHT_BAD
        console.print(f"  {u} -> {v}: {weight:.4f}", style=style)
    console.print("-" * 20, style=C_NEUTRAL)

# --- Data Collection & Metrics ---

@dataclass
class BenchmarkResult:
    architecture: str
    task_name: str
    success: bool
    duration_s: float
    llm_calls: int
    total_tokens: int
    final_answer: str
    notes: str = ""

class LLMCallTracker:
    """A simple class to track LLM API calls and token usage."""
    def __init__(self):
        self.calls = 0
        self.total_tokens = 0
        self._original_call_llm = UltimateReVALAgent._call_llm

    def reset(self):
        self.calls = 0
        self.total_tokens = 0

    def start_tracking(self):
        """Monkey-patch the agent's LLM call method to intercept calls."""
        async def tracked_call_llm(agent_instance, messages):
            self.calls += 1
            # Properly await the original coroutine
            response_message = await self._original_call_llm(agent_instance, messages)
            
            # Simple token estimation for now
            # A real implementation would parse the usage object from the response
            prompt_tokens = sum(agent_instance._tokens(m.get("content", "")) for m in messages)
            completion_tokens = agent_instance._tokens(response_message.get("content", ""))
            self.total_tokens += prompt_tokens + completion_tokens
            
            return response_message

        UltimateReVALAgent._call_llm = tracked_call_llm

    def stop_tracking(self):
        """Restore the original method."""
        UltimateReVALAgent._call_llm = self._original_call_llm

# --- Architectures to Benchmark ---

# 1. Baseline: A rigid, sequential chain of agents
async def run_baseline_chain(prompt: str, agents: list) -> str:
    """Simulates a hardcoded multi-agent system."""
    current_payload = prompt
    for agent in agents:
        print_info(f"Baseline: Passing to {agent.persona_prompt.split('.')[0]}")
        current_payload = await agent.chat(current_payload)
        if "error" in current_payload.lower() or "don't know" in current_payload.lower():
            return current_payload # Chain breaks on failure
    return current_payload

# 2. Our System: SynapseReVAL Network
async def run_synapsereval_network(prompt: str, net: SynapseNetwork, router_tools) -> str:
    """Runs the full, dynamic SynapseReVAL network."""
    # Set the prompt for the router's tool
    router_tools.original_prompt = prompt
    
    # Run the network
    history = await net.run(
        max_cycles=12,
        entry_neuron="SemanticRouter",
        initial_prompt=prompt,
        neuron_delay=0.2
    )
    
    final_answer = "No final answer produced."
    for entry in reversed(history):
        if entry.get("neuron") == "Finalizer":
            final_answer = entry.get("payload", "Finalizer fired but had no payload.")
            break
    return final_answer

# --- Main Benchmark Script ---

async def main():
    print_header("SynapseReVAL Benchmark Suite")
    
    log_file = os.path.join('logs', f"benchmark_run_{time.strftime('%Y%m%d-%H%M%S')}.log")
    print_info(f"Detailed logs will be saved to: {log_file}")
    
    os.environ["HTTP_REFERER"] = "https://synapsereval.local"
    os.environ["X_TITLE"] = "SynapseREval"

    all_results: list[BenchmarkResult] = []
    
    # --- Define Tasks ---
    tasks = [
        {
            "name": "Known Math (Division)",
            "prompt": "Please calculate 512 divided by 16.",
            "expected": "32"
        },
        {
            "name": "Novel Math (Exponentiation)",
            "prompt": "What is 8 to the power of 3?",
            "expected": "512"
        },
        {
            "name": "Ambiguous Task (Philosophy)",
            "prompt": "What is the true nature of a number?",
            "expected": "philosophical" # Success is qualitative here
        }
    ]

    # --- Initialize Agent Factory ---
    def agent_factory(persona_prompt: str, include_defaults: bool = True):
        return UltimateReVALAgent(
            model="deepseek/deepseek-chat-v3-0324:free",
            tool_support=True, temperature=0.1, debug=False, # Disable verbose debug for benchmark
            persona_prompt=persona_prompt,
            register_default_tools=include_defaults
        )
    
    # --- Initialize LLM Call Tracker ---
    tracker = LLMCallTracker()
    tracker.start_tracking() # This will patch the LLM call method for all agents

    # --- Benchmark Loop ---
    for task in tasks:
        print_header(f"Running Task: {task['name']}")
        print_info(f"Prompt: {task['prompt']}")
        
        # --- Architecture 1: Baseline Run ---
        arch_name = "Baseline Chain"
        print_info(f"Testing Architecture: {arch_name}")
        baseline_notes = ""
        try:
            tracker.reset() # Reset tracker before run
            
            # Setup baseline agents
            planner_agent_base = agent_factory("You are a planner. Break down the task.")
            # Give the worker a pre-built divide tool, but no exponentiation tool
            @tool("Divides two numbers.")
            def divide(a: float, b: float): return a / b
            worker_agent_base = agent_factory("You are a worker. Execute the plan using your tools.", include_defaults=False)
            worker_agent_base.register_tool(divide)
            
            start_time = time.time()
            final_answer_base = await run_baseline_chain(
                task['prompt'], 
                [planner_agent_base, worker_agent_base]
            )
            duration = time.time() - start_time
            success = task['expected'] in final_answer_base
            if not success: baseline_notes = "Failed to produce correct answer or use/build required tool."

            all_results.append(BenchmarkResult(
                architecture=arch_name, task_name=task['name'], success=success,
                duration_s=duration, llm_calls=tracker.calls, total_tokens=tracker.total_tokens,
                final_answer=final_answer_base, notes=baseline_notes
            ))
        except Exception as e:
            duration = time.time() - start_time
            all_results.append(BenchmarkResult(
                architecture=arch_name, task_name=task['name'], success=False,
                duration_s=duration, llm_calls=tracker.calls, total_tokens=tracker.total_tokens,
                final_answer=f"CRITICAL FAILURE: {e}", notes="System crashed."
            ))

        # --- Architecture 2: SynapseReVAL Run ---
        arch_name = "SynapseReVAL"
        print_info(f"Testing Architecture: {arch_name}")
        synapse_notes = ""
        try:
            tracker.reset() # Reset tracker before run
            
            # Setup SynapseReVAL network
            net = SynapseNetwork(learning_rate=0.2)
            semantic_router_tools = SemanticRouterTools(net)
            
            router_agent = agent_factory("...", include_defaults=False)
            router_agent.register_tool(semantic_router_tools.send_weighted_signals)
            
            coder_agent = agent_factory("You are a highly specialized Coder...")
            
            net.add_neuron("SemanticRouter", Neuron(router_agent, threshold=0.5))
            net.add_neuron("Coder", Neuron(coder_agent, threshold=0.8))
            net.add_neuron("Philosopher", Neuron(agent_factory("You are a Philosopher..."), threshold=0.8))
            net.add_neuron("Finalizer", Neuron(agent_factory("You are a Finalizer..."), threshold=1.0))

            net.connect("Coder", "Philosopher", weight=-2.5, plastic=False)
            net.connect("Philosopher", "Coder", weight=-2.5, plastic=False)
            net.connect("Coder", "Finalizer", weight=1.5, plastic=True)
            net.connect("Philosopher", "Finalizer", weight=1.0, plastic=True)

            start_time = time.time()
            final_answer_synapse = await run_synapsereval_network(
                task['prompt'], net, semantic_router_tools
            )
            duration = time.time() - start_time
            
            if task['expected'] == 'philosophical':
                success = "philosophy" in final_answer_synapse.lower() or "abstract" in final_answer_synapse.lower()
            else:
                success = task['expected'] in final_answer_synapse
            
            if "power" in coder_agent.tools: synapse_notes = "Tool 'power' created autonomously."

            all_results.append(BenchmarkResult(
                architecture=arch_name, task_name=task['name'], success=success,
                duration_s=duration, llm_calls=tracker.calls, total_tokens=tracker.total_tokens,
                final_answer=final_answer_synapse, notes=synapse_notes
            ))
            print_weights(net) # Show learning
        except Exception as e:
            duration = time.time() - start_time
            all_results.append(BenchmarkResult(
                architecture=arch_name, task_name=task['name'], success=False,
                duration_s=duration, llm_calls=tracker.calls, total_tokens=tracker.total_tokens,
                final_answer=f"CRITICAL FAILURE: {e}", notes="System crashed."
            ))

    # --- Stop Tracking and Report Results ---
    tracker.stop_tracking()
    print_header("Benchmark Results")

    # Console Table
    table = Table(title="Architecture Performance Comparison")
    table.add_column("Architecture", style="cyan", no_wrap=True)
    table.add_column("Task", style="magenta")
    table.add_column("Success", justify="center")
    table.add_column("Time (s)", justify="right", style="green")
    table.add_column("LLM Calls", justify="right", style="yellow")
    table.add_column("Tokens", justify="right", style="yellow")
    table.add_column("Notes", style="blue")

    for r in all_results:
        success_str = f"[{C_SUCCESS}]✔ Yes[/]" if r.success else f"[{C_FAILURE}]✖ No[/]"
        table.add_row(
            r.architecture, r.task_name, success_str, f"{r.duration_s:.2f}",
            str(r.llm_calls), str(r.total_tokens), r.notes
        )
    console.print(table)

    # CSV Output
    csv_file = "benchmark_results.csv"
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(BenchmarkResult.__annotations__.keys())
            for r in all_results:
                writer.writerow([
                    r.architecture, r.task_name, r.success, r.duration_s,
                    r.llm_calls, r.total_tokens, r.final_answer, r.notes
                ])
        print_success(f"Results saved to {csv_file}")
    except Exception as e:
        print_failure(f"Could not save CSV file: {e}")

if __name__ == "__main__":
    asyncio.run(main())