#!/usr/bin/env python3
"""
Tests for the Big Five Linguistic Adaptation Model module.
"""

import pytest
import numpy as np
from source.bigfivemodel import TemperamentProfile, TraitState, TraitSpectrum


def test_balanced_profile_creation():
    """Test creating a balanced profile with default values."""
    balanced_profile = TemperamentProfile()
    # Verify all traits are at balanced intensity
    for spectrum_name in [
        "extraversion", "agreeableness", "conscientiousness", 
        "emotional_stability", "openness"
    ]:
        trait_state = getattr(balanced_profile, spectrum_name)
        assert trait_state.intensity == 0.0
        assert trait_state.spectrum.value == spectrum_name


def test_custom_profile_creation():
    """Test creating a profile with specific trait intensities."""
    custom_profile = TemperamentProfile(
        extraversion=TraitState(spectrum=TraitSpectrum.EXTRAVERSION, intensity=0.6),
        agreeableness=TraitState(spectrum=TraitSpectrum.AGREEABLENESS, intensity=0.3),
        openness=TraitState(spectrum=TraitSpectrum.OPENNESS, intensity=-0.3)
    )
    
    # Check specific traits were set correctly
    assert custom_profile.extraversion.intensity == 0.6
    assert custom_profile.extraversion.trait_name == "Exuberant"
    
    assert custom_profile.agreeableness.intensity == 0.3
    assert custom_profile.agreeableness.trait_name == "Cooperative"
    
    assert custom_profile.openness.intensity == -0.3
    assert custom_profile.openness.trait_name == "Practical"
    
    # Check default traits are balanced
    assert custom_profile.conscientiousness.intensity == 0.0


def test_create_from_trait_names():
    """Test creating a profile from trait names."""
    named_profile = TemperamentProfile.create_from_trait_names(
        extraversion="Exuberant",
        conscientiousness="Perfectionist",
        emotional_stability="Sensitive"
    )
    
    # Check intensities match the trait names
    assert named_profile.extraversion.intensity == 0.6
    assert named_profile.conscientiousness.intensity == 1.0
    assert named_profile.emotional_stability.intensity == -0.3
    
    # Check names are set correctly
    assert named_profile.extraversion.trait_name == "Exuberant"
    assert named_profile.conscientiousness.trait_name == "Perfectionist"
    assert named_profile.emotional_stability.trait_name == "Sensitive"


def test_get_dominant_traits():
    """Test getting dominant traits from a profile."""
    custom_profile = TemperamentProfile(
        extraversion=TraitState(spectrum=TraitSpectrum.EXTRAVERSION, intensity=0.6),
        agreeableness=TraitState(spectrum=TraitSpectrum.AGREEABLENESS, intensity=0.3),
        openness=TraitState(spectrum=TraitSpectrum.OPENNESS, intensity=-0.3)
    )
    
    dominant = custom_profile.get_dominant_traits(2)
    
    # Check that we got exactly 2 traits
    assert len(dominant) == 2
    
    # Check that the most dominant trait is extraversion
    assert dominant[0]["spectrum"] == "extraversion"
    assert dominant[0]["trait"] == "Exuberant"
    assert dominant[0]["intensity"] == 0.6
    
    # Check that the second most dominant trait depends on absolute intensity
    # Both agreeableness and openness have absolute intensity 0.3, so either could be second
    assert abs(dominant[1]["intensity"]) == 0.3


def test_to_vector_and_from_vector():
    """Test converting a profile to a vector and back."""
    custom_profile = TemperamentProfile(
        extraversion=TraitState(spectrum=TraitSpectrum.EXTRAVERSION, intensity=0.6),
        agreeableness=TraitState(spectrum=TraitSpectrum.AGREEABLENESS, intensity=0.3),
        openness=TraitState(spectrum=TraitSpectrum.OPENNESS, intensity=-0.3)
    )
    
    # Convert to vector
    vector = custom_profile.to_vector()
    
    # Check vector dimensions
    assert len(vector) == 5
    assert vector[0] == 0.6  # extraversion
    assert vector[1] == 0.3  # agreeableness
    assert vector[4] == -0.3  # openness
    
    # Create profile from vector
    reconstructed = TemperamentProfile.from_vector(vector)
    
    # Check reconstructed profile
    assert reconstructed.extraversion.intensity == 0.6
    assert reconstructed.agreeableness.intensity == 0.3
    assert reconstructed.openness.intensity == -0.3


def test_blend_profiles():
    """Test blending two personality profiles."""
    profile1 = TemperamentProfile(
        extraversion=TraitState(spectrum=TraitSpectrum.EXTRAVERSION, intensity=0.6),
        agreeableness=TraitState(spectrum=TraitSpectrum.AGREEABLENESS, intensity=0.3)
    )
    
    profile2 = TemperamentProfile(
        extraversion=TraitState(spectrum=TraitSpectrum.EXTRAVERSION, intensity=0.0),
        agreeableness=TraitState(spectrum=TraitSpectrum.AGREEABLENESS, intensity=1.0),
        openness=TraitState(spectrum=TraitSpectrum.OPENNESS, intensity=-1.0)
    )
    
    # Blend with 30% of profile2
    blended = profile1.blend_with(profile2, weight=0.3)
    
    # Check blended values
    assert pytest.approx(blended.extraversion.intensity, 0.01) == 0.42  # 0.6 * 0.7 + 0.0 * 0.3
    assert pytest.approx(blended.agreeableness.intensity, 0.01) == 0.51  # 0.3 * 0.7 + 1.0 * 0.3
    assert pytest.approx(blended.openness.intensity, 0.01) == -0.3  # 0.0 * 0.7 + (-1.0) * 0.3
    
    # Verify trait names match the expected values for the blended intensity
    assert blended.extraversion.trait_name == "Sociable"  # Closest to 0.42 is 0.3 which maps to "Sociable"
    assert blended.agreeableness.trait_name == "Compassionate"  # Closest to 0.51 is 0.6 which maps to "Compassionate"
    assert blended.openness.trait_name == "Practical"  # Closest to -0.3 is -0.3 which maps to "Practical"