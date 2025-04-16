#!/usr/bin/env python3
"""
TemperamentAgent class for integrating Big Five personality traits with LLM interactions.

This module provides a class that manages the temperament of an LLM-based agent
and generates appropriate system prompts based on the personality profile.
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Union, Callable

# Import Temperament model
from source.bigfivemodel import TemperamentProfile, TraitState, TraitSpectrum
from source.temperament_prompt_generator import generate_temperament_prompt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TemperamentAgent:
    """A class for managing an LLM agent with personality traits using the Big Five model."""
    
    def __init__(
        self, 
        initial_profile: Optional[TemperamentProfile] = None,
        base_system_prompt: str = "You are a helpful assistant."
    ):
        """
        Initialize the TemperamentAgent.
        
        Args:
            initial_profile: Initial personality profile (defaults to balanced)
            base_system_prompt: Base system prompt to append personality guidelines to
        """
        self.profile = initial_profile if initial_profile else TemperamentProfile()
        self.baseline_profile = self.profile.model_copy(deep=True)
        self.base_system_prompt = base_system_prompt
        self.last_update_time = time.time()
        self.interaction_history = []
        
        # Generate initial prompt
        self.current_prompt = self._generate_prompt()
        
        logger.info("Temperament agent initialized")
    
    def _generate_prompt(self) -> str:
        """Generate a system prompt based on the current personality profile."""
        personality_guidelines = generate_temperament_prompt(self.profile)
        return f"{self.base_system_prompt}\n\n{personality_guidelines}"
    
    def update_temperament(self, trait_adjustments: Optional[Dict[str, float]] = None) -> None:
        """
        Update the agent's personality traits based on direct adjustments.
        
        Args:
            trait_adjustments: Direct adjustments to trait spectra (e.g., {"extraversion": 0.2})
        """
        # Apply direct trait adjustments
        if trait_adjustments:
            for spectrum_name, adjustment in trait_adjustments.items():
                self.profile.adjust_trait(spectrum_name, adjustment)
                logger.debug(f"Adjusted {spectrum_name} by {adjustment}")
        
        # Update the prompt based on the new temperament state
        self.current_prompt = self._generate_prompt()
        self.last_update_time = time.time()
        logger.info("Updated temperament state and prompt")
    
    def get_system_prompt(self) -> str:
        """Get the current system prompt with personality guidelines."""
        return self.current_prompt
    
    def record_interaction(self, 
                          user_message: str, 
                          agent_response: str, 
                          detected_user_traits: Optional[Dict[str, float]] = None,
                          agent_traits: Optional[Dict[str, float]] = None) -> None:
        """
        Record an interaction for history and analysis.
        
        Args:
            user_message: The message from the user
            agent_response: The agent's response
            detected_user_traits: Detected traits in the user message
            agent_traits: The agent's traits during response generation
        """
        interaction = {
            "timestamp": time.time(),
            "user_message": user_message,
            "agent_response": agent_response,
            "detected_user_traits": detected_user_traits,
            "agent_traits": agent_traits or self.profile.to_vector().tolist()
        }
        self.interaction_history.append(interaction)
        logger.debug("Recorded interaction")
    
    def set_baseline_profile(self, profile: TemperamentProfile) -> None:
        """
        Set a new baseline personality profile.
        
        Args:
            profile: The new baseline personality profile
        """
        self.baseline_profile = profile.model_copy(deep=True)
        logger.info("Updated baseline personality profile")
    
    def save_state(self, filepath: str) -> None:
        """
        Save the agent's current state to a file.
        
        Args:
            filepath: Path to save the state file
        """
        state = {
            "current_profile": self.profile.model_dump(),
            "baseline_profile": self.baseline_profile.model_dump(),
            "base_system_prompt": self.base_system_prompt,
            "last_update_time": self.last_update_time,
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"Saved agent state to {filepath}")
        except Exception as e:
            logger.error(f"Error saving agent state: {e}")
    
    @classmethod
    def load_state(cls, filepath: str) -> "TemperamentAgent":
        """
        Load an agent state from a file.
        
        Args:
            filepath: Path to the state file
            
        Returns:
            A TemperamentAgent with the loaded state
        """
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Create profiles from the state
            current_profile = TemperamentProfile.model_validate(state["current_profile"])
            baseline_profile = TemperamentProfile.model_validate(state["baseline_profile"])
            
            # Create the agent
            agent = cls(
                initial_profile=current_profile,
                base_system_prompt=state["base_system_prompt"]
            )
            
            # Set additional state properties
            agent.baseline_profile = baseline_profile
            agent.last_update_time = state["last_update_time"]
            
            logger.info(f"Loaded agent state from {filepath}")
            return agent
            
        except Exception as e:
            logger.error(f"Error loading agent state: {e}")
            return cls()  # Return default agent if loading fails


# Example usage
if __name__ == "__main__":
    # Create an initial profile - a friendly, sociable assistant
    initial_profile = TemperamentProfile.create_from_trait_names(
        extraversion="Sociable",
        agreeableness="Cooperative"
    )
    
    # Create the agent
    agent = TemperamentAgent(
        initial_profile=initial_profile,
        base_system_prompt="You are a helpful AI assistant designed to provide information and assistance."
    )
    
    # Print the initial prompt
    print("=== Initial System Prompt ===")
    print(agent.get_system_prompt())
    
    # Simulate a trait adjustment
    print("\n=== Adjusting Traits ===")
    agent.update_temperament(trait_adjustments={
        "conscientiousness": 0.6,
        "openness": 0.3
    })
    
    # Print the updated prompt
    print("\n=== Updated System Prompt ===")
    print(agent.get_system_prompt())
    
    # Save the agent state
    agent.save_state("agent_state.json")
    print("\nSaved agent state to agent_state.json")