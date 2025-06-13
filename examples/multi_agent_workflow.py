#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Agent Workflow Example using SynapseNetwork

This example creates a complex network of specialized agents that work together
to solve a research and analysis task through a bio-inspired neural architecture.
"""
import asyncio
import os
import sys
import time
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.llm import UltimateReVALAgent
from core.synapses import SynapseNetwork, Neuron
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure brain file path
BRAIN_FILE = os.path.join(os.path.dirname(__file__), '..', 'brains', 'research_workflow.json')

class ResearchAgent(UltimateReVALAgent):
    """Research agent that specializes in finding information."""
    
    def __init__(self, name, model="gpt-4", debug=False):
        super().__init__(model=model, debug=debug)
        self.name = name
        self.set_persona_prompt(
            f"You are {name}, a specialized research agent that excels at finding "
            f"relevant information on topics. You focus solely on gathering facts "
            f"and data without analysis. Be thorough but concise."
        )
        
        # Register custom tools
        self.register_tool(self.search_web, "Search the web for information on a topic")
        
    async def search_web(self, query: str) -> str:
        """Simulate searching the web for information."""
        # In a real implementation, this would connect to a search API
        await asyncio.sleep(1)  # Simulate network delay
        return f"Research results for '{query}': [Simulated web search results would appear here]"

class AnalysisAgent(UltimateReVALAgent):
    """Analysis agent that specializes in analyzing information."""
    
    def __init__(self, name, model="gpt-4", debug=False):
        super().__init__(model=model, debug=debug)
        self.name = name
        self.set_persona_prompt(
            f"You are {name}, a specialized analysis agent that excels at analyzing "
            f"and synthesizing information. You identify patterns, draw connections, "
            f"and extract insights from raw data. Be analytical and precise."
        )

class WritingAgent(UltimateReVALAgent):
    """Writing agent that specializes in creating content."""
    
    def __init__(self, name, model="gpt-4", debug=False):
        super().__init__(model=model, debug=debug)
        self.name = name
        self.set_persona_prompt(
            f"You are {name}, a specialized writing agent that excels at creating "
            f"well-structured, engaging content. You transform ideas and analysis "
            f"into clear, coherent text optimized for the target audience."
        )

class PlanningAgent(UltimateReVALAgent):
    """Planning agent that coordinates the workflow."""
    
    def __init__(self, name, model="gpt-4", debug=False):
        super().__init__(model=model, debug=debug)
        self.name = name
        self.set_persona_prompt(
            f"You are {name}, a specialized planning agent that excels at coordinating "
            f"complex workflows. You break down tasks, assign responsibilities, and "
            f"ensure the overall process runs smoothly. Be strategic and organized."
        )

async def setup_network(debug=False):
    """Set up the neural network with specialized agents."""
    # Create the network
    network = SynapseNetwork()
    
    # Create specialized agents
    planner = PlanningAgent("PlannerAgent", debug=debug)
    researcher1 = ResearchAgent("PrimaryResearcher", debug=debug)
    researcher2 = ResearchAgent("SecondaryResearcher", debug=debug)
    analyst1 = AnalysisAgent("DataAnalyst", debug=debug)
    analyst2 = AnalysisAgent("InsightAnalyst", debug=debug)
    writer = WritingAgent("ContentWriter", debug=debug)
    
    # Add agents as neurons to the network
    network.add_neuron("planner", planner, role="planner", threshold=0.5)
    network.add_neuron("researcher1", researcher1, role="researcher", threshold=0.7)
    network.add_neuron("researcher2", researcher2, role="researcher", threshold=0.7)
    network.add_neuron("analyst1", analyst1, role="analyst", threshold=0.6)
    network.add_neuron("analyst2", analyst2, role="analyst", threshold=0.6)
    network.add_neuron("writer", writer, role="writer", threshold=0.5)
    
    # Connect neurons with synapses (directed connections with weights)
    network.connect("planner", "researcher1", weight=1.0, plastic=True)
    network.connect("planner", "researcher2", weight=0.8, plastic=True)
    network.connect("researcher1", "analyst1", weight=1.0, plastic=True)
    network.connect("researcher2", "analyst1", weight=0.9, plastic=True)
    network.connect("analyst1", "analyst2", weight=1.0, plastic=True)
    network.connect("analyst2", "writer", weight=1.0, plastic=True)
    network.connect("writer", "planner", weight=0.7, plastic=True)  # Feedback loop
    
    return network

async def run_research_workflow(topic, network):
    """Run a complete research workflow on a topic."""
    print(f"\n🧠 Starting research workflow on topic: {topic}\n")
    
    # Initial spike to the planner
    initial_prompt = f"""
    We need to research, analyze, and create a report on the topic: "{topic}".
    Please coordinate the research workflow by:
    1. Breaking down the research areas
    2. Assigning specific aspects to researchers
    3. Planning how the analysis should be structured
    4. Outlining what the final report should contain
    """
    
    # Start the workflow with the planner
    response = await network.activate("planner", initial_prompt)
    print(f"🧠 Initial planning complete: {len(response)} chars\n")
    
    # Run for several cycles to complete the workflow
    for cycle in range(1, 6):
        print(f"\n🧠 Workflow Cycle {cycle} 🧠")
        
        # Let the network continue processing based on synaptic connections
        # The network will automatically route messages between agents
        responses = await network.cycle()
        
        for neuron_id, response in responses.items():
            print(f"  • {neuron_id}: {len(response)} chars")
        
        # Allow time to observe the process
        await asyncio.sleep(1)
    
    # Get the final output from the writer
    final_report = network.get_memory("writer")
    
    print("\n🧠 Workflow Complete 🧠")
    print(f"Final report length: {len(final_report)} chars")
    
    # Save the brain state for visualization
    network.save(BRAIN_FILE)
    print(f"Network brain saved to: {BRAIN_FILE}")
    
    return final_report

async def main():
    # Set up the neural network
    debug_mode = True  # Enable debug mode for detailed logging
    network = await setup_network(debug=debug_mode)
    
    # Define a research topic
    research_topic = "The impact of artificial neural networks on modern AI applications"
    
    # Run the workflow
    final_report = await run_research_workflow(research_topic, network)
    
    # Display a summary of the final report
    print("\n🧠 Final Report Summary 🧠")
    print(f"Topic: {research_topic}")
    print(f"Report length: {len(final_report)} characters")
    print(f"Report preview: {final_report[:200]}...")
    
    # Visualize the network if the visualization tool is available
    visualize_script = os.path.join(os.path.dirname(__file__), '..', 'tools', 'visualize_network.py')
    if os.path.exists(visualize_script):
        print("\n🧠 Generating network visualization...")
        os.system(f"python {visualize_script} {BRAIN_FILE} --output network_viz.png --history-output history_viz.png")
        print("Visualization files created: network_viz.png, history_viz.png")

if __name__ == "__main__":
    asyncio.run(main()) 