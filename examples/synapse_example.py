#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example of using SynapseNetwork with UltimateReVALAgent for a multi-agent task.
This example creates a simple network of specialized agents to solve a problem.
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
load_dotenv(".env.local")

async def main():
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a timestamped log file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(logs_dir, f"synapse_example_{timestamp}.log")
    
    print(f"Debug logs will be saved to: {log_file}")
    
    # Set HTTP_REFERER and X_TITLE for OpenRouter
    os.environ["HTTP_REFERER"] = "https://synapsereval.local"
    os.environ["X_TITLE"] = "SynapseREval"

    # Agent factory function to create specialized agents
    def agent_factory(name: str, role: str):
        # Create personalized prompts for each agent role
        if role == "researcher":
            prompt = f"You are {name}, a research specialist. Your job is to analyze questions and break them down into key research areas. Be concise and focus on identifying the core topics that need to be investigated."
        elif role == "analyst":
            prompt = f"You are {name}, a data analyst. Your job is to take research topics and provide factual information about them. Be precise and informative."
        elif role == "writer":
            prompt = f"You are {name}, a professional writer. Your job is to take analytical information and craft it into a well-structured, easy-to-understand response. Focus on clarity and engagement."
        else:
            prompt = f"You are {role} named {name}. Respond succinctly."
            
        return UltimateReVALAgent(
            model="deepseek/deepseek-chat-v3-0324:free",
            tool_support=True,
            temperature=0.2,
            max_model_tokens=16_000,
            max_response_tokens=2_048,
            persona_prompt=prompt,
            debug=True,
            debug_log_file=log_file,
        )

    # Create the neural network
    net = SynapseNetwork(learning_rate=0.05)
    
    # Add specialized neurons
    net.add_neuron("Researcher", Neuron(agent_factory("Researcher", "researcher"), threshold=1.0, role="researcher"))
    net.add_neuron("Analyst", Neuron(agent_factory("Analyst", "analyst"), threshold=1.0, role="analyst"))
    net.add_neuron("Writer", Neuron(agent_factory("Writer", "writer"), threshold=1.0, role="writer"))

    # Connect neurons in a linear workflow
    # Researcher -> Analyst -> Writer
    net.connect("Researcher", "Analyst", weight=1.5)
    net.connect("Analyst", "Writer", weight=1.2)
    
    # Optional feedback loop for iterative improvement
    # Writer -> Researcher (with lower weight)
    net.connect("Writer", "Researcher", weight=0.3)

    # Define the question to be answered
    question = "What are the key differences between renewable and non-renewable energy sources, and which is more economically viable in the long term?"
    
    print(f"\nQuestion: {question}\n")
    print("Processing through neural network...\n")
    
    # Run the network with the question
    history = await net.run(
        max_cycles=6,
        entry_neuron="Researcher",
        initial_prompt=question
    )
    
    # Extract the final answer from the Writer
    final_answer = None
    for entry in reversed(history):
        if entry["neuron"] == "Writer":
            final_answer = entry["payload"]
            break
    
    # Print the result
    if final_answer:
        print("\nFinal Answer:\n")
        print(final_answer)
    else:
        print("\nNo final answer was produced. Check the logs for details.")
    
    # Save the network state
    brain_path = Path(os.path.join(os.path.dirname(__file__), '..', 'brains', f"energy_question_{timestamp}.json"))
    brain_path.parent.mkdir(exist_ok=True)
    net.save(str(brain_path))
    
    print(f"\nNetwork state saved to: {brain_path}")
    print(f"Full debug logs available in: {log_file}")

if __name__ == "__main__":
    asyncio.run(main()) 