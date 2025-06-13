# SynapseNetwork Integration with UltimateReVALAgent

This document explains how the SynapseNetwork neural architecture integrates with the UltimateReVALAgent class to create a bio-inspired multi-agent orchestration system.

## Overview

The SynapseNetwork is a neural architecture that treats each agent as a neuron and message pathways as weighted synapses. It provides a biologically-inspired approach to multi-agent orchestration with features like:

- **Per-Neuron Memory**: Local key-value store and rolling transcript for each neuron
- **STDP Plasticity**: Spike-timing dependent synaptic weight updates
- **Multi-Modal Spikes**: Support for text, image, audio payloads
- **Attentional Routing**: Softmax attention over outgoing synapses
- **Persistence**: JSON serialization for long-lived "brains"

## Integration with UltimateReVALAgent

The SynapseNetwork uses UltimateReVALAgent instances as the "brains" of each neuron. The integration works as follows:

1. **Agent Factory**: The system creates UltimateReVALAgent instances with specific roles and names
2. **Neuron Wrapping**: Each agent is wrapped in a `Neuron` class that manages:
   - Input accumulation (inbox)
   - Firing threshold logic
   - Memory storage
   - Response generation

3. **Network Structure**: Neurons are connected via weighted, directional synapses
4. **Spike Propagation**: When a neuron fires, its output is routed to connected neurons based on synapse weights
5. **Plasticity**: Synapse weights change based on spike timing (STDP)

## Usage Example

```python
async def main():
    # Create agent factory function
    def agent_factory(name: str, role: str):
        prompt = f"You are {role} named {name}. Respond succinctly."
        return UltimateReVALAgent(
            model="deepseek/deepseek-chat-v3-0324:free",
            tool_support=True,
            temperature=0.2,
            max_model_tokens=16_000,
            max_response_tokens=2_048,
            persona_prompt=prompt,
            debug=True,
            debug_log_file="agent_log.txt",
        )

    # Create network
    net = SynapseNetwork(learning_rate=0.05)
    
    # Add neurons
    net.add_neuron("Planner", Neuron(agent_factory("Planner", "planner"), threshold=1.0, role="planner"))
    net.add_neuron("Executor", Neuron(agent_factory("Executor", "executor"), threshold=1.0, role="executor"))
    net.add_neuron("Finalizer", Neuron(agent_factory("Finalizer", "finalizer"), threshold=1.0, role="finalizer"))

    # Connect neurons
    net.connect("Planner", "Executor", weight=1.2)
    net.connect("Executor", "Finalizer", weight=1.0)
    net.connect("Finalizer", "Planner", weight=0.4)

    # Run network with initial prompt
    await net.run(max_cycles=8, entry_neuron="Planner", initial_prompt="Your question here")
    
    # Save network state
    net.save("brain.json")
```

## Debugging

The SynapseNetwork leverages the debug capabilities of UltimateReVALAgent:

1. Each neuron's agent can have its own debug log file
2. The SynapseNetwork logs neuron firing, spike routing, and STDP updates
3. Network state is saved to a JSON file for inspection

## Known Issues

1. **OpenRouter API Errors**: Occasionally the OpenRouter API returns 500 errors, which can stop the network propagation
2. **Character Encoding**: The JSON serialization may have encoding issues with non-ASCII characters

## Best Practices

1. Use appropriate thresholds for neurons (usually 1.0 is a good default)
2. Set reasonable learning rates (0.02-0.05) for STDP plasticity
3. Limit maximum cycles to prevent infinite loops (6-10 cycles is usually sufficient)
4. Design the network topology carefully to achieve the desired information flow
5. Use the debug logs to understand the network's behavior

## Future Improvements

1. Add support for more modalities (images, audio)
2. Implement more sophisticated routing mechanisms
3. Add dynamic topology modifications via LLM tools
4. Improve error handling for API failures
5. Add visualization tools for network state and activity 