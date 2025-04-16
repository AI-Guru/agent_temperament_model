#!/usr/bin/env python3
"""
Example usage of the Temperament Prompt Generator.

This script demonstrates how to create different personality profiles
and generate corresponding prompts for LLM agents.
"""

import json
import os
from source.bigfivemodel import TemperamentProfile, TraitState, TraitSpectrum
from source.temperament_prompt_generator import generate_temperament_prompt


def save_prompt_to_file(prompt, filename):
    """Save a generated prompt to a file."""
    with open(filename, 'w') as f:
        f.write(prompt)
    print(f"Saved prompt to {filename}")


def main():
    # Create output directory if it doesn't exist
    os.makedirs("prompts", exist_ok=True)
    
    # Example 1: Extraverted and Agreeable Assistant
    print("\n=== Example 1: Extraverted and Agreeable Assistant ===")
    extraverted_profile = TemperamentProfile(
        extraversion=TraitState(spectrum=TraitSpectrum.EXTRAVERSION, intensity=0.6),  # Exuberant
        agreeableness=TraitState(spectrum=TraitSpectrum.AGREEABLENESS, intensity=0.6)  # Compassionate
    )
    extraverted_prompt = generate_temperament_prompt(extraverted_profile)
    print(extraverted_prompt)
    save_prompt_to_file(extraverted_prompt, "prompts/extraverted_agreeable_assistant.txt")
    
    # Example 2: Analytical and Stable Assistant
    print("\n=== Example 2: Analytical and Stable Assistant ===")
    analytical_profile = TemperamentProfile(
        conscientiousness=TraitState(spectrum=TraitSpectrum.CONSCIENTIOUSNESS, intensity=0.6),  # Meticulous
        emotional_stability=TraitState(
            spectrum=TraitSpectrum.EMOTIONAL_STABILITY, 
            intensity=0.6  # Imperturbable
        ),
        openness=TraitState(
            spectrum=TraitSpectrum.OPENNESS, 
            intensity=0.3  # Exploratory
        )
    )
    analytical_prompt = generate_temperament_prompt(analytical_profile)
    print(analytical_prompt)
    save_prompt_to_file(analytical_prompt, "prompts/analytical_stable_assistant.txt")
    
    # Example 3: Balanced Communicator
    print("\n=== Example 3: Balanced Communicator ===")
    balanced_profile = TemperamentProfile.create_from_trait_names(
        extraversion="Balanced",
        agreeableness="Cooperative",
        conscientiousness="Adaptable",
        emotional_stability="Resilient",
        openness="Receptive"
    )
    balanced_prompt = generate_temperament_prompt(balanced_profile)
    print(balanced_prompt)
    save_prompt_to_file(balanced_prompt, "prompts/balanced_communicator.txt")
    
    # Example 4: Creative Innovator
    print("\n=== Example 4: Creative Innovator ===")
    creative_profile = TemperamentProfile()
    creative_profile.adjust_trait("openness", 1.0)  # Visionary
    creative_profile.adjust_trait("extraversion", 0.3)  # Sociable
    creative_profile.adjust_trait("conscientiousness", -0.3)  # Flexible
    creative_prompt = generate_temperament_prompt(creative_profile)
    print(creative_prompt)
    save_prompt_to_file(creative_prompt, "prompts/creative_innovator.txt")
    
    # Example 5: Reserved Specialist
    print("\n=== Example 5: Reserved Specialist ===")
    reserved_profile = TemperamentProfile(
        extraversion=TraitState(spectrum=TraitSpectrum.EXTRAVERSION, intensity=-0.3),  # Reserved
        conscientiousness=TraitState(spectrum=TraitSpectrum.CONSCIENTIOUSNESS, intensity=0.6),  # Meticulous
        openness=TraitState(
            spectrum=TraitSpectrum.OPENNESS, 
            intensity=-0.3  # Practical
        )
    )
    reserved_prompt = generate_temperament_prompt(reserved_profile)
    print(reserved_prompt)
    save_prompt_to_file(reserved_prompt, "prompts/reserved_specialist.txt")
    
    # Create a JSON file with all the profiles for reference
    profiles = {
        "extraverted_agreeable": extraverted_profile.model_dump(),
        "analytical_stable": analytical_profile.model_dump(),
        "balanced_communicator": balanced_profile.model_dump(),
        "creative_innovator": creative_profile.model_dump(),
        "reserved_specialist": reserved_profile.model_dump()
    }
    
    with open("prompts/temperament_profiles.json", 'w') as f:
        json.dump(profiles, f, indent=2)
    print("\nSaved all temperament profiles to prompts/temperament_profiles.json")


if __name__ == "__main__":
    main()