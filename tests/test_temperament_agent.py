#!/usr/bin/env python3
"""
Tests for the TemperamentAgent module.
"""

import pytest
import json
import time
import os
from typing import Dict
from source.bigfivemodel import TemperamentProfile
from source.temperament_agent import TemperamentAgent


def test_agent_initialization():
    """Test creating a TemperamentAgent with default and custom profiles."""
    # Test with default profile
    default_agent = TemperamentAgent()
    assert default_agent.profile is not None
    assert default_agent.profile.extraversion.intensity == 0.0
    
    # Test with custom profile
    initial_profile = TemperamentProfile.create_from_trait_names(
        extraversion="Exuberant",
        agreeableness="Cooperative"
    )
    custom_agent = TemperamentAgent(
        initial_profile=initial_profile,
        base_system_prompt="Custom prompt."
    )
    
    assert custom_agent.profile.extraversion.intensity == 0.6
    assert custom_agent.profile.agreeableness.intensity == 0.3
    assert custom_agent.base_system_prompt == "Custom prompt."


def test_generate_system_prompt():
    """Test generating a system prompt from the personality profile."""
    initial_profile = TemperamentProfile.create_from_trait_names(
        extraversion="Exuberant",
        agreeableness="Cooperative"
    )
    agent = TemperamentAgent(
        initial_profile=initial_profile,
        base_system_prompt="You are a helpful AI assistant."
    )
    
    # Get the prompt
    prompt = agent.get_system_prompt()
    
    # Check that it contains the base system prompt
    assert "You are a helpful AI assistant." in prompt
    
    # Check that it contains trait info
    assert "Exuberant" in prompt
    assert "Cooperative" in prompt


def test_trait_state_update_direct():
    """Test updating trait state with direct adjustments."""
    agent = TemperamentAgent()
    
    # Initial state should be balanced
    assert agent.profile.extraversion.intensity == 0.0
    
    # Update traits directly
    agent.update_temperament(trait_adjustments={"extraversion": 0.5})
    
    # Check that trait was updated
    assert agent.profile.extraversion.intensity == 0.5
    
    # Another update should add to the existing value
    agent.update_temperament(trait_adjustments={"extraversion": 0.2})
    # The value should be approximately 0.7 (may have small floating-point differences)
    assert pytest.approx(agent.profile.extraversion.intensity, 0.001) == 0.7


def test_trait_analyzer_integration():
    """Test integration with a trait analyzer function."""
    agent = TemperamentAgent()
    
    # Create a sample trait analyzer function
    def simple_trait_analyzer(text: str) -> Dict[str, float]:
        """A very simple trait analyzer that looks for key words."""
        traits = {
            "extraversion": 0.0,
            "agreeableness": 0.0,
            "openness": 0.0
        }
        
        if "social" in text.lower():
            traits["extraversion"] = 0.5
        if "critical" in text.lower():
            traits["agreeableness"] = -0.5
            
        return traits
    
    # Analyze user message manually and update traits
    user_message = "I'm a very social person, but I can be quite critical of new ideas."
    traits = simple_trait_analyzer(user_message)
    
    # Update temperament with the analyzed traits
    agent.update_temperament(trait_adjustments=traits)
    
    # Check that traits were updated
    assert agent.profile.extraversion.intensity > 0  # Moved toward high extraversion
    assert agent.profile.agreeableness.intensity < 0  # Moved toward low agreeableness


def test_record_interaction():
    """Test recording an interaction."""
    agent = TemperamentAgent()
    
    # Record an interaction
    agent.record_interaction(
        user_message="Hello",
        agent_response="Hi there",
        detected_user_traits={"extraversion": 0.3},
        agent_traits={"extraversion": 0.2}
    )
    
    # Check that interaction was recorded
    assert len(agent.interaction_history) == 1
    interaction = agent.interaction_history[0]
    assert interaction["user_message"] == "Hello"
    assert interaction["agent_response"] == "Hi there"
    assert interaction["detected_user_traits"] == {"extraversion": 0.3}
    assert interaction["agent_traits"] == {"extraversion": 0.2}


def test_save_and_load_state(tmp_path):
    """Test saving and loading agent state."""
    # Create an agent with custom settings
    initial_profile = TemperamentProfile.create_from_trait_names(
        extraversion="Exuberant",
        agreeableness="Cooperative"
    )
    agent = TemperamentAgent(
        initial_profile=initial_profile,
        base_system_prompt="Custom prompt."
    )
    
    # Save state to a temporary file
    filepath = tmp_path / "test_state.json"
    agent.save_state(filepath)
    
    # Verify file exists
    assert os.path.exists(filepath)
    
    # Load state into a new agent
    loaded_agent = TemperamentAgent.load_state(filepath)
    
    # Verify loaded state matches original
    assert loaded_agent.profile.extraversion.intensity == agent.profile.extraversion.intensity
    assert loaded_agent.profile.agreeableness.intensity == agent.profile.agreeableness.intensity
    assert loaded_agent.base_system_prompt == agent.base_system_prompt