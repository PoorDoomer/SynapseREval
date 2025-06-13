
# How to Use SynapseReVAL

This guide provides a comprehensive walkthrough for using the **SynapseReVAL** framework. It covers the core concepts, setup, practical examples, and best practices for building adaptive multi-agent systems.

## The Core Philosophy: From Rigid Chains to Dynamic Brains

Traditional multi-agent systems often use a hard-coded, sequential chain: `Agent A -> Agent B -> Agent C`. This is predictable but brittle; it fails if any step is unexpected.

**SynapseReVAL** uses a different model inspired by the brain. You are not building a chain; you are building a **small, artificial brain** where:
-   **Agents are Neurons:** Each `UltimateReVALAgent` is a specialized processing unit.
-   **Information is a Spike:** Data flows through the network as a "spike" with an activation level.
-   **Connections are Synapses:** The pathways between neurons have weights that can be positive (excitatory) or negative (inhibitory).
-   **Behavior is Emergent:** The system's overall behavior arises from the interaction of these components, allowing it to adapt, compete, and learn.

## Quickstart: Building Your First Network

Here is the essential code structure for creating and running a SynapseReVAL network.

```python
import asyncio
from core.synapses import SynapseNetwork, Neuron
from core.llm import UltimateReVALAgent

async def main():
    # 1. Define an Agent Factory
    # This function creates our 'neuron' brains.
    def agent_factory(name: str, persona: str, has_defaults: bool = True):
        return UltimateReVALAgent(
            model="deepseek/deepseek-chat-v3-0324:free",
            persona_prompt=persona,
            register_default_tools=has_defaults,
            debug=True,
            debug_log_file=f"logs/{name.lower()}_agent.log"
        )

    # 2. Create the Network
    # The learning_rate controls how quickly synapse weights change.
    net = SynapseNetwork(learning_rate=0.1)

    # 3. Create Specialized Agents (Neurons)
    # Each neuron gets an agent with a specific persona and a firing threshold.
    planner_agent = agent_factory("Planner", "You are a master planner.")
    worker_agent = agent_factory("Worker", "You are a diligent worker who executes plans.")
    
    net.add_neuron("Planner", Neuron(planner_agent, threshold=1.0))
    net.add_neuron("Worker", Neuron(worker_agent, threshold=1.0))

    # 4. Connect Neurons with Synapses
    # Create a simple excitatory pathway.
    net.connect("Planner", "Worker", weight=1.5, plastic=True)

    # 5. Run the Network
    # Inject an initial prompt into the 'Planner' neuron to start the process.
    initial_task = "Develop a three-step plan to analyze market trends."
    history = await net.run(
        entry_neuron="Planner",
        initial_prompt=initial_task,
        max_cycles=5
    )

    # 6. (Optional) Save the Learned Brain
    net.save("my_first_brain.json")

if __name__ == "__main__":
    asyncio.run(main())
```

## Advanced Usage: The Winner-Take-All Circuit

This is where the true power of SynapseReVAL shines. Instead of a simple chain, we can create a competitive circuit where the best agent for the job dynamically wins.

This example uses **lateral inhibition**, where two specialist neurons actively suppress each other.

```python
# --- Setup ---
net = SynapseNetwork(learning_rate=0.2)

# Create a Coder and a Philosopher agent
coder_agent = agent_factory("Coder", "You are a Coder. You solve concrete problems.")
philo_agent = agent_factory("Philosopher", "You are a Philosopher. You ponder abstract ideas.")

net.add_neuron("Coder", Neuron(coder_agent, threshold=0.8))
net.add_neuron("Philosopher", Neuron(philo_agent, threshold=0.8))
net.add_neuron("Finalizer", Neuron(agent_factory("Finalizer", "You present the final answer."), threshold=1.0))

# --- The Circuit ---
# 1. Lateral Inhibition (Negative Weights)
# When one fires, it sends a strong negative signal to the other.
net.connect("Coder", "Philosopher", weight=-2.5, plastic=False)
net.connect("Philosopher", "Coder", weight=-2.5, plastic=False)

# 2. Excitatory Output Pathways (Positive Weights)
# The winner sends its result to the Finalizer. These are plastic to learn.
net.connect("Coder", "Finalizer", weight=1.5, plastic=True)
net.connect("Philosopher", "Finalizer", weight=1.0, plastic=True)

# --- Execution ---
# A router would inject a prompt with a high activation for the Coder
net.inject(dst="Coder", payload="Calculate 25 * 16", weight=2.0)
net.inject(dst="Philosopher", payload="Calculate 25 * 16", weight=0.1) # Low relevance

# In the subsequent run, the Coder will fire, inhibit the Philosopher, and solve the problem.
await net.run(max_cycles=5)
```

## Best Practices & Design Patterns

Building with SynapseReVAL is more like architecture than programming. Here are key principles to follow.

#### 1. **Isolate Specialized Functions**
An agent should do one thing well. Don't create a "do-everything" agent. Instead, create a `Coder`, a `Writer`, a `Data_Analyst`, etc. This modularity is key to robust systems.

-   **Bad:** An agent with the persona "You are an expert who can code, write, and analyze data."
-   **Good:** Three separate agents: a `Coder`, a `Writer`, and a `Data_Analyst`, connected in a logical workflow.

#### 2. **Control Agent Capabilities Architecturally**
The most reliable way to control an agent is to limit its tools. A `Router` agent should not have access to the `create_and_test_tool`.

```python
# Good Practice: Create a constrained agent
router_persona = "You are a router. Your only job is to call the `route_task` tool."
# Create this agent WITHOUT the default tools
router_agent = agent_factory(
    "Router", 
    router_persona, 
    include_defaults=False # This flag comes from llm.py's __init__
)
# Now, ONLY register the specific tools it needs
router_agent.register_tool(my_routing_tool)
```

#### 3. **Use Inhibition for Competition and Control**
Negative weights are your most powerful tool for creating intelligent control flows.
-   **Winner-Take-All:** Have specialists mutually inhibit each other to ensure only the most activated one proceeds.
-   **Quality Gate:** An `Evaluator` neuron could send an inhibitory signal back to a `Worker` if the quality is poor, forcing the `Worker` to retry without proceeding down the chain.

#### 4. **Fortify Personas with Guard Rails**
LLMs have a strong bias to be "helpful." You must be explicit to force them into a specialized role. Use strong, clear constraints.

-   **Weak Persona:** "You are a Coder. Please use tools to solve the problem."
-   **Strong Persona (Good Practice):**
    > "You are a specialized Coder agent. You are **FORBIDDEN** from answering directly. Your **ONLY** purpose is to use tools. Your process **MUST** be: 1. Analyze request. 2. Check tools. 3. If tool exists, call it. 4. If not, call `create_and_test_tool`. Your response **MUST** be only a tool call."

#### 5. **Tune Network Parameters Carefully**
-   **`threshold`**: A neuron's willingness to fire. Higher thresholds (e.g., `1.5`) require more consensus from multiple input neurons. Lower thresholds (e.g., `0.5`) make them more "reactive."
-   **`learning_rate`**: How quickly the brain learns. Higher rates (e.g., `0.2`) are good for fast adaptation in testing. Lower rates (e.g., `0.02`) are better for stable learning in production.
-   **`max_cycles`**: A critical safety valve to prevent infinite loops, especially in networks with feedback connections. `5-12` cycles is typically a safe range.

## Debugging and Inspection

-   **Agent-Level Logs**: Each `UltimateReVALAgent` can have its own log file specified during creation. This is essential for inspecting the "internal thoughts" (the ReVAL loop) of a specific neuron.
-   **Network-Level Logs**: The `SynapseNetwork` itself logs high-level events: neuron firings, spike activations, inhibitory signals, and STDP weight changes.
-   **The "Brain" File**: Use `net.save("brain_state.json")` at any point to dump the entire network structure, including the final learned weights of all synapses. This is invaluable for seeing if your network is learning as expected.

## Known Issues and Future Directions

-   **API Errors**: The system relies on external LLM APIs. Occasional `500` errors from services like OpenRouter can interrupt a run. The `UltimateReVALAgent` includes a retry mechanism, but persistent failures will halt the network.
-   **Prompt Brittleness**: As demonstrated in the benchmarks, agent behavior is highly sensitive to the persona prompt. Continuous refinement and testing of prompts is required.
-   **Future Improvements**: The architecture is designed for expansion. Key future directions include adding more sophisticated routing mechanisms (e.g., a router that dynamically adjusts activation weights), integrating vector databases for long-term memory, and allowing a "supervisor" agent to dynamically rewire the network topology.