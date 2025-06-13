#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script for UltimateReVALAgent."""

import sys
import os

# Add the parent directory to the path so we can import from core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.llm import UltimateReVALAgent
from core.tools import Tools

def main():
    """Test the UltimateReVALAgent initialization."""
    print("Initializing agent...")
    
    # Mock the API key for testing
    os.environ["OPENAI_API_KEY"] = "sk-mock-key-for-testing"
    tools = Tools(db_path="core/databasevf.db")
    agent = UltimateReVALAgent(debug=True, tool_support=True, model="deepseek/deepseek-chat-v3-0324:free")
    agent.register_tools_from_instance(tools)
    print("Agent initialized successfully!")
    
    # Print the system prompt to verify it's set correctly
    print("\nSystem prompt:")
    print(agent.system_prompt[:200] + "..." if len(agent.system_prompt) > 200 else agent.system_prompt)
    
    # Print the registered tools
    print("\nRegistered tools:")
    for tool_name in agent.tools:
        print(f"- {tool_name}")

if __name__ == "__main__":
    main() 