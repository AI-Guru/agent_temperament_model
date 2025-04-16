#!/usr/bin/env python3
"""
Tests for the temperament prompt generator module.
"""

import pytest
import json
import os
from source.bigfivemodel import TemperamentProfile, TraitState, TraitSpectrum
from source.temperament_prompt_generator import (
    generate_temperament_prompt,
    extract_primary_secondary_traits,
    format_trait_description,
    get_linguistic_features,
    generate_adaptation_rules,
    create_temperament_profile_from_json
)


def test_extract_primary_secondary_traits():
    """Test extracting primary and secondary traits from a profile."""
    # Create a profile with varying intensities
    profile = TemperamentProfile(
        extraversion=TraitState(spectrum=TraitSpectrum.EXTRAVERSION, intensity=0.8),
        agreeableness=TraitState(spectrum=TraitSpectrum.AGREEABLENESS, intensity=0.3),
        openness=TraitState(spectrum=TraitSpectrum.OPENNESS, intensity=-0.6)
    )
    
    result = extract_primary_secondary_traits(profile)
    
    # Check structure
    assert "primary" in result
    assert "secondary" in result
    
    # Primary should be extraversion (highest absolute intensity)
    assert result["primary"]["spectrum"] == "extraversion"
    # The trait at 0.8 intensity is Effusive
    assert result["primary"]["trait"] == "Effusive"
    assert result["primary"]["intensity"] == 0.8
    
    # Secondary should be openness (second highest absolute intensity)
    assert result["secondary"]["spectrum"] == "openness"
    assert result["secondary"]["trait"] == "Conventional"
    assert result["secondary"]["intensity"] == -0.6


def test_format_trait_description():
    """Test formatting trait data into natural language descriptions."""
    trait_dict = {
        "spectrum": "extraversion",
        "trait": "Exuberant",
        "intensity": 0.6
    }
    
    result = format_trait_description(trait_dict)
    
    # Check result
    assert result["trait"] == "Exuberant"
    assert result["intensity"] == "strong"
    assert result["direction"] == "high"
    assert result["raw_intensity"] == 0.6
    assert result["spectrum"] == "extraversion"
    
    # Test with negative intensity
    trait_dict = {
        "spectrum": "agreeableness",
        "trait": "Antagonistic",
        "intensity": -0.6
    }
    
    result = format_trait_description(trait_dict)
    assert result["intensity"] == "strong"
    assert result["direction"] == "low"
    assert result["spectrum"] == "agreeableness"


def test_get_linguistic_features():
    """Test generating linguistic feature suggestions based on trait."""
    trait_dict = {
        "trait": "Exuberant",
        "intensity": "strong",
        "direction": "high",
        "raw_intensity": 0.6,
        "spectrum": "extraversion"
    }
    
    features = get_linguistic_features(trait_dict)
    
    # Check result contains expected guidance for extraversion
    assert "enthusiastic expressions" in features.lower()
    assert "varied sentence structure" in features.lower()
    
    # Test with different trait
    trait_dict = {
        "trait": "Antagonistic",
        "intensity": "moderate",
        "direction": "low",
        "raw_intensity": -0.6,
        "spectrum": "agreeableness"
    }
    
    features = get_linguistic_features(trait_dict)
    
    # Check result contains expected guidance for low agreeableness
    assert "skeptical questioning" in features.lower()
    assert "debate ideas" in features.lower()


def test_generate_adaptation_rules():
    """Test generating adaptation rules based on traits."""
    primary = {
        "trait": "Exuberant", 
        "intensity": "strong",
        "direction": "high",
        "raw_intensity": 0.6,
        "spectrum": "extraversion"
    }
    
    secondary = {
        "trait": "Cooperative",
        "intensity": "moderate",
        "direction": "high",
        "raw_intensity": 0.3,
        "spectrum": "agreeableness"
    }
    
    rules = generate_adaptation_rules(primary, secondary)
    
    # Check result is a string
    assert isinstance(rules, str)
    
    # Check it contains the common adaptation points
    assert "When helping with problem-solving" in rules
    
    # Check it contains specific adaptations for the traits
    assert "When discussing technical topics" in rules  # Extraversion adaptation
    assert "When addressing conflicts or disagreements" in rules  # Agreeableness adaptation


def test_create_temperament_profile_from_json():
    """Test creating a temperament profile from JSON data."""
    json_data = {
        "extraversion": 0.6,
        "agreeableness": -0.3,
        "openness": 0.0
    }
    
    profile = create_temperament_profile_from_json(json_data)
    
    # Check profile was created correctly
    assert profile.extraversion.intensity == 0.6
    assert profile.agreeableness.intensity == -0.3
    assert profile.openness.intensity == 0.0


def test_generate_temperament_prompt():
    """Test generating a complete temperament prompt."""
    # Create a profile
    profile = TemperamentProfile(
        extraversion=TraitState(spectrum=TraitSpectrum.EXTRAVERSION, intensity=0.6),
        agreeableness=TraitState(spectrum=TraitSpectrum.AGREEABLENESS, intensity=0.3)
    )
    
    prompt = generate_temperament_prompt(profile)
    
    # Check prompt is a string
    assert isinstance(prompt, str)
    
    # Check it contains the key sections
    assert "You are an assistant with the following personality traits" in prompt
    assert "Primary:" in prompt
    assert "Secondary:" in prompt
    assert "Express these traits naturally" in prompt
    assert "Adapt your personality expression" in prompt
    
    # Check it mentions the traits
    assert "Exuberant" in prompt
    assert "Cooperative" in prompt