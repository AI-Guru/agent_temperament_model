#!/usr/bin/env python3
"""
Big Five Linguistic Adaptation Model implementation using Pydantic v2.

This module provides a comprehensive implementation of the Big Five personality model,
which defines five personality trait spectra, each with varying intensity levels
from negative to positive. The model allows for quantification, representation,
and manipulation of personality traits.
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from typing import Dict, List, Literal, Optional, Union, Tuple
from enum import Enum
import numpy as np
from typing_extensions import Annotated


class TraitSpectrum(str, Enum):
    """The five personality trait spectra defined in the Big Five model"""
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EMOTIONAL_STABILITY = "emotional_stability"
    OPENNESS = "openness"


# Standard intensity levels for traits in Big Five model
INTENSITY_VALUES = {
    -1.0: "extreme_negative", 
    -0.6: "moderate_negative",
    -0.3: "mild_negative",
    0.0: "balanced",
    0.3: "mild_positive",
    0.6: "moderate_positive",
    1.0: "extreme_positive"
}


# Define the trait labels for each spectrum with their corresponding intensities
TRAIT_MAPPINGS = {
    TraitSpectrum.EXTRAVERSION: {
        -1.0: "Reclusive",
        -0.6: "Withdrawn",
        -0.3: "Reserved",
        0.0: "Balanced",
        0.3: "Sociable",
        0.6: "Exuberant",
        1.0: "Effusive",
    },
    TraitSpectrum.AGREEABLENESS: {
        -1.0: "Hostile",
        -0.6: "Antagonistic",
        -0.3: "Challenging",
        0.0: "Objective",
        0.3: "Cooperative",
        0.6: "Compassionate",
        1.0: "Selfless",
    },
    TraitSpectrum.CONSCIENTIOUSNESS: {
        -1.0: "Chaotic",
        -0.6: "Disorganized",
        -0.3: "Flexible",
        0.0: "Adaptable",
        0.3: "Methodical",
        0.6: "Meticulous",
        1.0: "Perfectionist",
    },
    TraitSpectrum.EMOTIONAL_STABILITY: {
        -1.0: "Turbulent",
        -0.6: "Volatile",
        -0.3: "Sensitive",
        0.0: "Temperate",
        0.3: "Resilient",
        0.6: "Imperturbable",
        1.0: "Stoic",
    },
    TraitSpectrum.OPENNESS: {
        -1.0: "Rigid",
        -0.6: "Conventional",
        -0.3: "Practical",
        0.0: "Receptive",
        0.3: "Exploratory",
        0.6: "Innovative",
        1.0: "Visionary",
    },
}


class TraitState(BaseModel):
    """Represents the state of a specific trait within a spectrum"""
    spectrum: TraitSpectrum
    intensity: Annotated[float, Field(ge=-1.0, le=1.0)]
    
    @property
    def trait_name(self) -> str:
        """Get the trait name based on intensity value"""
        # Find the closest predefined intensity level
        intensity_levels = list(TRAIT_MAPPINGS[self.spectrum].keys())
        closest_intensity = min(intensity_levels, key=lambda x: abs(x - self.intensity))
        return TRAIT_MAPPINGS[self.spectrum][closest_intensity]

    @classmethod
    def from_trait_name(cls, spectrum: TraitSpectrum, trait_name: str) -> "TraitState":
        """Create a TraitState from a spectrum and trait name"""
        # Find the intensity for the given trait name
        for intensity, name in TRAIT_MAPPINGS[spectrum].items():
            if name.lower() == trait_name.lower():
                return cls(spectrum=spectrum, intensity=float(intensity))
        
        # If not found, raise an error
        raise ValueError(f"Trait '{trait_name}' not found in spectrum {spectrum}")


class TemperamentProfile(BaseModel):
    """Complete personality profile using Big Five model"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    extraversion: TraitState = Field(
        default_factory=lambda: TraitState(spectrum=TraitSpectrum.EXTRAVERSION, intensity=0.0)
    )
    agreeableness: TraitState = Field(
        default_factory=lambda: TraitState(spectrum=TraitSpectrum.AGREEABLENESS, intensity=0.0)
    )
    conscientiousness: TraitState = Field(
        default_factory=lambda: TraitState(spectrum=TraitSpectrum.CONSCIENTIOUSNESS, intensity=0.0)
    )
    emotional_stability: TraitState = Field(
        default_factory=lambda: TraitState(spectrum=TraitSpectrum.EMOTIONAL_STABILITY, intensity=0.0)
    )
    openness: TraitState = Field(
        default_factory=lambda: TraitState(spectrum=TraitSpectrum.OPENNESS, intensity=0.0)
    )
    
    @model_validator(mode='after')
    def validate_spectrum_assignments(self) -> "TemperamentProfile":
        """Validate that each trait state is assigned to the correct spectrum"""
        # Ensure each trait state is assigned to its corresponding spectrum
        for field_name, expected_spectrum in [
            ("extraversion", TraitSpectrum.EXTRAVERSION),
            ("agreeableness", TraitSpectrum.AGREEABLENESS),
            ("conscientiousness", TraitSpectrum.CONSCIENTIOUSNESS),
            ("emotional_stability", TraitSpectrum.EMOTIONAL_STABILITY),
            ("openness", TraitSpectrum.OPENNESS),
        ]:
            trait_state = getattr(self, field_name)
            if trait_state.spectrum != expected_spectrum:
                trait_state.spectrum = expected_spectrum
        return self
    
    def get_dominant_traits(self, n: int = 2) -> List[Dict]:
        """Get the n most dominant traits (by absolute intensity)"""
        all_states = [
            self.extraversion,
            self.agreeableness,
            self.conscientiousness,
            self.emotional_stability,
            self.openness
        ]
        
        # Sort by absolute intensity (descending)
        sorted_states = sorted(all_states, key=lambda s: abs(s.intensity), reverse=True)
        
        return [
            {
                "spectrum": state.spectrum.value,
                "trait": state.trait_name,
                "intensity": state.intensity
            }
            for state in sorted_states[:n]
        ]
    
    def to_vector(self) -> np.ndarray:
        """Convert the personality profile to a 5-dimensional vector"""
        return np.array([
            self.extraversion.intensity,
            self.agreeableness.intensity,
            self.conscientiousness.intensity,
            self.emotional_stability.intensity,
            self.openness.intensity
        ])
    
    @classmethod
    def from_vector(cls, vector: List[float]) -> "TemperamentProfile":
        """Create a TemperamentProfile from a 5-dimensional vector"""
        if len(vector) != 5:
            raise ValueError("Vector must have exactly 5 dimensions")
        
        return cls(
            extraversion=TraitState(spectrum=TraitSpectrum.EXTRAVERSION, intensity=vector[0]),
            agreeableness=TraitState(spectrum=TraitSpectrum.AGREEABLENESS, intensity=vector[1]),
            conscientiousness=TraitState(spectrum=TraitSpectrum.CONSCIENTIOUSNESS, intensity=vector[2]),
            emotional_stability=TraitState(
                spectrum=TraitSpectrum.EMOTIONAL_STABILITY, 
                intensity=vector[3]
            ),
            openness=TraitState(spectrum=TraitSpectrum.OPENNESS, intensity=vector[4])
        )
    
    @classmethod
    def create_from_trait_names(cls, **traits) -> "TemperamentProfile":
        """
        Create a profile using trait names
        Example: TemperamentProfile.create_from_trait_names(extraversion="Sociable", openness="Innovative")
        """
        profile = cls()
        
        for field_name, trait_name in traits.items():
            if hasattr(profile, field_name):
                spectrum = getattr(profile, field_name).spectrum
                setattr(profile, field_name, TraitState.from_trait_name(spectrum, trait_name))
        
        return profile
    
    def adjust_trait(self, spectrum_name: str, adjustment: float) -> None:
        """
        Adjust a trait's intensity by the specified amount, keeping within bounds
        
        Args:
            spectrum_name: Name of the trait spectrum to adjust
            adjustment: Amount to adjust the intensity (-1.0 to 1.0)
        """
        if hasattr(self, spectrum_name):
            trait_state = getattr(self, spectrum_name)
            new_intensity = min(max(trait_state.intensity + adjustment, -1.0), 1.0)
            setattr(self, spectrum_name, TraitState(
                spectrum=trait_state.spectrum,
                intensity=new_intensity
            ))
    
    def blend_with(self, other_profile: "TemperamentProfile", weight: float = 0.5) -> "TemperamentProfile":
        """
        Blend this profile with another profile using the specified weight
        
        Args:
            other_profile: Another TemperamentProfile to blend with
            weight: Weight of the other profile (0.0 to 1.0), where 0.0 is all this profile
                   and 1.0 is all other profile
        
        Returns:
            A new TemperamentProfile that is a blend of the two profiles
        """
        if not 0.0 <= weight <= 1.0:
            raise ValueError("Weight must be between 0.0 and 1.0")
        
        this_vector = self.to_vector()
        other_vector = other_profile.to_vector()
        
        # Linear interpolation
        blended_vector = (1 - weight) * this_vector + weight * other_vector
        
        return TemperamentProfile.from_vector(blended_vector.tolist())


# Example usage
if __name__ == "__main__":
    # Create a profile with default balanced values
    balanced_profile = TemperamentProfile()
    print(f"Balanced profile: all traits at intensity 0.0")
    
    # Create a profile with specific trait intensities
    custom_profile = TemperamentProfile(
        extraversion=TraitState(spectrum=TraitSpectrum.EXTRAVERSION, intensity=0.6),
        agreeableness=TraitState(spectrum=TraitSpectrum.AGREEABLENESS, intensity=0.3),
        openness=TraitState(spectrum=TraitSpectrum.OPENNESS, intensity=-0.3)
    )
    print(f"\nCustom profile traits:")
    print(f"Extraversion: {custom_profile.extraversion.trait_name}")
    print(f"Agreeableness: {custom_profile.agreeableness.trait_name}")
    print(f"Openness: {custom_profile.openness.trait_name}")
    
    # Create a profile from trait names
    named_profile = TemperamentProfile.create_from_trait_names(
        extraversion="Exuberant",
        conscientiousness="Perfectionist",
        emotional_stability="Sensitive"
    )
    print(f"\nProfile created from trait names:")
    print(f"Extraversion intensity: {named_profile.extraversion.intensity}")
    print(f"Conscientiousness intensity: {named_profile.conscientiousness.intensity}")
    print(f"Emotional Stability intensity: {named_profile.emotional_stability.intensity}")
    
    # Get dominant traits
    dominant = custom_profile.get_dominant_traits(2)
    print(f"\nDominant traits: {dominant}")
    
    # Convert to and from vector
    vector = custom_profile.to_vector()
    print(f"\nTrait vector: {vector}")
    
    # Blend two profiles
    blended_profile = custom_profile.blend_with(named_profile, weight=0.3)
    print(f"\nBlended profile traits:")
    print(f"Extraversion: {blended_profile.extraversion.trait_name}")
    print(f"Agreeableness: {blended_profile.agreeableness.trait_name}")
    print(f"Conscientiousness: {blended_profile.conscientiousness.trait_name}")
    print(f"Emotional Stability: {blended_profile.emotional_stability.trait_name}")
    print(f"Openness: {blended_profile.openness.trait_name}")