
# SynapseReVAL: A Bio-Inspired Multi-Agent System

**SynapseReVAL** is an advanced framework for building multi-agent AI systems that bridges the gap between rigid, programmed workflows and dynamic, emergent intelligence. It combines two powerful, synergistic concepts:

1.  **`SynapseNetwork`**: A bio-inspired orchestration layer where individual agents act as **neurons**. Their communication pathways are **synapses** that can learn and adapt over time through simulated neural plasticity.
2.  **`UltimateReVALAgent`**: A highly robust, self-correcting agent endowed with a cognitive loop for reasoning, verification, and adaptation (**ReVAL**). These agents can autonomously create their own tools to solve novel problems.

Together, they create a system where a flexible, adaptive **macro-structure** (the network) orchestrates highly intelligent and resilient **micro-units** (the agents), enabling the system to tackle complex, ambiguous, and unforeseen tasks.

  
*(Recommendation: Create a simple diagram showing a router, specialists with inhibitory connections, and a finalizer, and upload it to a host like Imgur to include here.)*

---

## Table of Contents

- [Core Concepts](#core-concepts)
- [Key Features](#key-features)
  - [SynapseNetwork (`synapses.py`)](#synapsenetwork-synapsespy)
  - [UltimateReVALAgent (`llm.py`)](#ultimaterevalagent-llmpy)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Usage](#usage)
  - [Defining a Network](#defining-a-network)
  - [Running the Definitive Benchmark](#running-the-definitive-benchmark)
- [Architectural Deep Dive](#architectural-deep-dive)
  - [The Neuron and the Spike](#the-neuron-and-the-spike)
  - [Excitatory vs. Inhibitory Synapses](#excitatory-vs-inhibitory-synapses)
  - [Learning with STDP](#learning-with-stdp)
  - [The Agent's Cognitive Loop (ReVAL)](#the-agents-cognitive-loop-reval)
- [Future Work & Contributing](#future-work--contributing)

---

## Core Concepts

Traditional multi-agent systems often rely on a central "manager" agent that dictates tasks in a rigid, top-down manner. This approach is brittle and fails when faced with unexpected problems.

**SynapseReVAL** proposes a different paradigm inspired by computational neuroscience:

-   **Agents as Neurons**: Each `UltimateReVALAgent` is a node in a graph. It has an **activation threshold** and fires only when it receives sufficient input from other neurons.
-   **Communication as Spikes**: Information is passed between neurons as **spikes**, which carry a payload (e.g., a text prompt) and an **activation** level (e.g., a relevance score).
-   **Workflows as Neural Pathways**: The connections between agents are **synapses** with numerical weights.
    -   **Excitatory Synapses (w > 0)**: A neuron firing across this synapse will *increase* the activation of the target neuron, encouraging it to fire.
    -   **Inhibitory Synapses (w < 0)**: A neuron firing across this synapse will *decrease* the activation of the target, suppressing it. This allows for **lateral inhibition**, where specialists can compete and the most relevant one "wins".
-   **Learning as Synaptic Plasticity**: The weights of excitatory synapses are not fixed. Using **Spike-Timing-Dependent Plasticity (STDP)**, the connection strength between two neurons is increased if they fire in a successful causal sequence, reinforcing effective pathways over time.

This creates a system where complex problem-solving behaviors can **emerge** from the interactions of simple, local rules, rather than being explicitly programmed.

## Key Features

### SynapseNetwork (`synapses.py`)

The orchestration layer that forms the "brain" of the system.

-   🧠 **Bio-Inspired Dynamics**: Models neural concepts like activation thresholds, excitation, and inhibition.
-   🏆 **Winner-Take-All Circuits**: Lateral inhibition allows for dynamic selection of the most relevant specialist for a given task.
-   📈 **Neurological Learning (STDP)**: Automatically reinforces successful problem-solving pathways by strengthening synaptic weights. "Neurons that fire together, wire together."
-   🧭 **Stochastic & Attentional Routing**: Uses a `softmax` function over excitatory synapse weights to probabilistically route information, paying more attention to stronger connections.
-   💾 **Persistence**: The entire state of the network, including learned weights and neuron memories, can be saved to and loaded from a JSON file.

### UltimateReVALAgent (`llm.py`)

The intelligent, resilient "neuron" that performs the actual work.

-   🔄 **ReVAL Cognitive Cycle**: Implements a **R**eason -> **V**erify -> **A**dapt -> **L**oop cycle, preventing agents from producing unverified or low-confidence outputs.
-   🛠️ **Autonomous Tool-Building**: Features the `create_and_test_tool` meta-tool. If an agent lacks a capability, it can write, test (in a **secure Docker sandbox**), and integrate a new Python tool for itself mid-task.
-   🎛️ **Architectural Constraint**: Supports initialization with or without a default set of tools, allowing for the creation of highly specialized agents with restricted capabilities (e.g., a `Router` that can *only* route).
-   🛡️ **Robust Guard Rails**: Built with modern best practices, including hierarchical context trimming, adaptive timeouts for tools, and graceful degradation for API features.
-   📝 **Rich Tooling**: Comes with a suite of built-in meta-tools for goal management (`update_goal_state`), short-term memory (`ScratchPad`), and self-reflection (`self_reflect_and_replan`).

## Getting Started

### Prerequisites

-   Python 3.10+
-   [Docker Desktop](https://www.docker.com/products/docker-desktop/): Required for the `create_and_test_tool` sandbox. Ensure the Docker daemon is running.
-   An API key for an OpenAI-compatible service (e.g., [OpenRouter.ai](https://openrouter.ai/)).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/PoorDoomer/SynapseREval.git
    cd SynapseREval
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file containing `openai`, `python-dotenv`, `networkx`, `numpy`, `rich`, and `docker`.)*

### Configuration

1.  Create a file named `.env.local` in the root of the project.
2.  Add your API key to this file:
    ```env
    # .env.local
    OPENROUTER_API_KEY="sk-or-your-key-here" (or use an even better LLM gateway the best in the world llmgateway.io ) 
    ```
3.  If you are using OpenRouter, you can also add the following for proper request headers:
    ```env
    HTTP_REFERER="https://your-app-url.com"
    X_TITLE="Your App Name"
    ```

## Usage

The `examples/` directory contains scripts that showcase the system's capabilities.

### Defining a Network

Building a network is intuitive. You define agents, add them as neurons, and connect them with synapses.

```python
# From examples/inhibitory_test.py

# 1. Create the network
net = SynapseNetwork(learning_rate=0.2)

# 2. Create specialized agents (neurons)
coder_agent = agent_factory("Coder", coder_prompt)
philosopher_agent = agent_factory("Philosopher", philosopher_prompt)

net.add_neuron("Coder", Neuron(coder_agent, threshold=0.8))
net.add_neuron("Philosopher", Neuron(philosopher_agent, threshold=0.8))

# 3. Connect them with excitatory and inhibitory synapses
# Lateral inhibition circuit
net.connect("Coder", "Philosopher", weight=-2.5) # Coder suppresses Philosopher
net.connect("Philosopher", "Coder", weight=-2.5) # Philosopher suppresses Coder

# Excitatory path to a final output neuron
net.connect("Coder", "Finalizer", weight=1.5, plastic=True) # This connection can learn!
```

### Running the Definitive Benchmark

The best way to see everything working together is to run the final benchmark script. This script tests the system's ability to route tasks, build tools autonomously, and learn from its successes.

**Ensure Docker is running**, then execute the following command from the project root:

```bash
python examples/benchmark.py
```

The script will run two architectures (a rigid baseline vs. SynapseReVAL) on three distinct tasks and output a summary table and a `benchmark_results.csv` file. This provides clear, quantitative evidence of the system's adaptive capabilities.

## Architectural Deep Dive

### The Neuron and the Spike

The `Neuron` is the fundamental computational unit. It contains an `inbox` that accumulates incoming `Spike` objects. A `Spike` has two key properties:
-   `payload`: The actual data being transmitted (e.g., "Calculate 512 / 16").
-   `activation`: A numerical value representing the strength or relevance of the signal.

Each cycle, the `Neuron` sums the `activation` of all spikes in its inbox. If this sum exceeds its `threshold`, the neuron "fires," activating its internal `UltimateReVALAgent` to process the payload.

### Excitatory vs. Inhibitory Synapses

The final activation of a spike depends on the synapse it crosses.
-   `Spike arrives at Neuron B = Spike's initial activation * weight of Synapse A->B`
-   If the weight is **positive (excitatory)**, it increases the target's activation, pushing it toward firing.
-   If the weight is **negative (inhibitory)**, it decreases the target's activation, suppressing it. This is the key to creating competitive "winner-take-all" dynamics.

### Learning with STDP

Spike-Timing-Dependent Plasticity is a simple but powerful learning rule from neuroscience.
-   **If Neuron A fires *just before* Neuron B, and this leads to a successful outcome, the A -> B synapse is strengthened.** This is because their firing was causally correlated in a useful way.
-   This is implemented by increasing the synapse weight based on an exponential decay of the time difference between the two firings. Pathways that are used frequently and effectively become stronger over time, optimizing the network's performance automatically.

### The Agent's Cognitive Loop (ReVAL)

The `UltimateReVALAgent` is not a simple function. It operates on a **Reason-Verify-Adapt-Loop**:
1.  **Reason:** It analyzes the prompt and its internal state to form a plan, which often involves calling a tool.
2.  **Verify:** It internally checks the output of its reasoning or tool use. Does the result seem correct? Is the confidence high? (In our code, this is implemented with the `simple_verifier` tool).
3.  **Adapt:** If verification fails or confidence is low, the agent can trigger a self-reflection step, critique its own plan, and adapt its approach for the next attempt.
4.  **Loop:** It continues this cycle until a satisfactory result is achieved or a limit is reached. This makes each agent incredibly robust against errors and hallucinations.

## Future Work & Contributing

SynapseReVAL is an experimental framework with a vast potential for expansion. Key areas for future research include:

-   **More Complex Topologies**: Exploring recurrent networks, deep networks, and modular brain-like structures.
-   **Advanced Routing**: Implementing a `Router` neuron that can send dynamically weighted signals based on a deeper semantic understanding of the prompt.
-   **Long-Term Memory**: Replacing the simple `NeuronMemory` with a vector database to give agents a persistent, searchable memory of past tasks.
-   **Dynamic Topology Changes**: Allowing a "meta-supervisor" agent to add/remove neurons and synapses at runtime in response to system performance.

Contributions are welcome! Please open an issue to discuss potential features or submit a pull request.
