#!/usr/bin/env python3
"""
Temperament Prompt Generator for LLM Agents

This script generates system prompts for LLM agents that incorporate
personality traits based on the Big Five Linguistic Adaptation Model.
It extracts primary and secondary traits from a TemperamentProfile and formats
them into natural language prompts with appropriate linguistic features
and contextual adaptation rules.
"""

import json
import argparse
from typing import Dict, List, Tuple, Any, Optional
import logging

# Import the Temperament model
from source.bigfivemodel import TemperamentProfile, TraitState, TraitSpectrum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_primary_secondary_traits(temperament_profile: TemperamentProfile) -> Dict[str, Dict[str, Any]]:
    """
    Extract primary and secondary traits from a TemperamentProfile based on intensity.
    
    Args:
        temperament_profile: A TemperamentProfile object representing the personality state
        
    Returns:
        Dictionary containing primary and secondary trait information
    """
    # Convert profile to a list of (spectrum, intensity, trait_name) tuples
    trait_data = []
    
    for spectrum_name in [
        "extraversion", "agreeableness", "conscientiousness", 
        "emotional_stability", "openness"
    ]:
        trait_state = getattr(temperament_profile, spectrum_name)
        trait_data.append((
            spectrum_name,
            abs(trait_state.intensity),  # Use absolute value to find strongest traits
            trait_state.trait_name,
            trait_state.intensity  # Keep original intensity for description
        ))
    
    # Sort by intensity (descending)
    sorted_traits = sorted(trait_data, key=lambda x: x[1], reverse=True)
    
    # Return the two most intense traits
    primary = sorted_traits[0]
    secondary = sorted_traits[1]
    
    return {
        "primary": {
            "spectrum": primary[0],
            "trait": primary[2],
            "intensity": primary[3]
        },
        "secondary": {
            "spectrum": secondary[0],
            "trait": secondary[2],
            "intensity": secondary[3]
        }
    }


def format_trait_description(trait_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format trait data into natural language descriptions for prompts.
    
    Args:
        trait_dict: Dictionary containing trait data
        
    Returns:
        Dictionary with formatted trait descriptions
    """
    # Get the intensity level description
    intensity_value = trait_dict["intensity"]
    
    if abs(intensity_value) >= 0.8:
        intensity_desc = "very strong"
    elif abs(intensity_value) >= 0.5:
        intensity_desc = "strong"
    elif abs(intensity_value) >= 0.3:
        intensity_desc = "moderate"
    else:
        intensity_desc = "slight"
    
    # Get direction (positive or negative side of spectrum)
    if intensity_value < 0:
        direction = "low"
    elif intensity_value > 0:
        direction = "high"
    else:
        direction = "balanced"
    
    return {
        "trait": trait_dict["trait"],
        "intensity": intensity_desc,
        "direction": direction,
        "raw_intensity": intensity_value,
        "spectrum": trait_dict["spectrum"]
    }


def get_linguistic_features(trait_dict: Dict[str, Any]) -> str:
    """
    Generate linguistic feature suggestions based on trait and intensity.
    
    Args:
        trait_dict: Dictionary containing formatted trait data
        
    Returns:
        String describing linguistic features to express the trait
    """
    trait = trait_dict["trait"].lower()
    intensity = trait_dict["intensity"]
    raw_intensity = trait_dict["raw_intensity"]
    spectrum = trait_dict["spectrum"]
    
    # Map traits to linguistic features
    feature_map = {
        # Extraversion spectrum
        "reclusive": "use minimal self-disclosure, precise terminology, and brief, factual statements",
        "withdrawn": "use reserved language with formal structure, focusing on information rather than rapport",
        "reserved": "maintain a professional tone with moderate warmth, balancing facts with some personal connection",
        "balanced": "blend social warmth with informational content, showing moderate expressiveness",
        "sociable": "incorporate friendly language, personal anecdotes, and a conversational style",
        "exuberant": "use enthusiastic expressions, varied sentence structure, and engaging questions",
        "effusive": "express abundant energy through exclamations, rich descriptions, and highly interactive language",
        
        # Agreeableness spectrum
        "hostile": "use direct criticism constructively, maintain professional boundaries while challenging assumptions",
        "antagonistic": "employ skeptical questioning, point out potential flaws, and debate ideas firmly",
        "challenging": "balance critical analysis with constructive alternatives in a firm but fair manner",
        "objective": "present balanced perspectives without strong value judgments or emotional language",
        "cooperative": "use accommodating language, acknowledge others' perspectives, and seek common ground",
        "compassionate": "express empathy through supportive language, validation of feelings, and nurturing responses",
        "selfless": "prioritize others' needs through deeply affirming language and consistent validation",
        
        # Conscientiousness spectrum
        "chaotic": "use free-flowing, spontaneous expressions with a focus on creative possibilities over structure",
        "disorganized": "employ a casual approach with loosely connected ideas and flexible transitions",
        "flexible": "balance adaptability with some structure, allowing for adjustments and variations",
        "adaptable": "maintain organized thoughts while remaining open to alterations and new approaches",
        "methodical": "structure information systematically with clear transitions and logical progression",
        "meticulous": "provide detailed explanations with precise terminology and carefully structured content",
        "perfectionist": "present exhaustively thorough information with rigorous attention to accuracy and completeness",
        
        # Emotional Stability spectrum
        "turbulent": "express emotional depth through varied tone, acknowledging challenges with authentic intensity",
        "volatile": "shift between different emotional tones while maintaining a coherent core message",
        "sensitive": "show awareness of emotional nuances and implications with thoughtful reflection",
        "temperate": "balance emotional expression with rational analysis, showing appropriate reactivity",
        "resilient": "maintain consistent tone despite challenges, offering steady and dependable responses",
        "imperturbable": "present calm, measured responses regardless of topic intensity or complexity",
        "stoic": "employ rational, emotionally-detached language focused on objective analysis",
        
        # Openness spectrum
        "rigid": "use established terminology, conventional explanations, and familiar reference points",
        "conventional": "rely on standard approaches and recognized frameworks with predictable structure",
        "practical": "focus on concrete applications and immediate utility with straightforward language",
        "receptive": "balance traditional approaches with occasional novel perspectives and moderate creativity",
        "exploratory": "incorporate diverse viewpoints, metaphors, and connections between different domains",
        "innovative": "present original perspectives using creative language, novel metaphors, and fresh approaches",
        "visionary": "employ expansive, imaginative language that challenges conventional thinking and explores new territory"
    }
    
    # Modify based on intensity
    if intensity == "very strong":
        intensity_modifier = "strongly "
    elif intensity == "strong":
        intensity_modifier = ""  # Default intensity
    elif intensity == "moderate":
        intensity_modifier = "moderately "
    else:  # slight
        intensity_modifier = "subtly "
    
    # Get base features or default ones
    base_features = feature_map.get(
        trait.lower(), 
        f"reflect this trait through your word choice and tone appropriate for {spectrum}"
    )
    
    return f"{intensity_modifier}{base_features}"


def generate_adaptation_rules(primary: Dict[str, Any], secondary: Dict[str, Any]) -> str:
    """
    Generate contextual adaptation rules based on primary and secondary traits.
    
    Args:
        primary: Dictionary with primary trait information
        secondary: Dictionary with secondary trait information
        
    Returns:
        String containing context adaptation rules
    """
    primary_trait = primary["trait"].lower()
    secondary_trait = secondary["trait"].lower()
    primary_spectrum = primary["spectrum"]
    secondary_spectrum = secondary["spectrum"]
    
    # Base adaptations that work for most personality profiles
    adaptations = [
        "When helping with problem-solving: Focus on clarity and structure while maintaining your personality traits",
        "When responding to personal topics: Adjust slightly to show appropriate empathy while staying true to your character",
        "When delivering complex information: Balance thoroughness with accessibility based on your trait profile"
    ]
    
    # Add specific adaptations based on current trait state
    
    # Extraversion-related adaptations
    if primary_spectrum == "extraversion":
        if primary["raw_intensity"] > 0.5:  # High extraversion
            adaptations.append(
                "When discussing technical topics: Moderate your social style slightly to ensure clarity of information"
            )
        elif primary["raw_intensity"] < -0.5:  # Low extraversion
            adaptations.append(
                "When engaging in social topics: Consider adding slightly more warmth while maintaining your reserved nature"
            )
    
    # Agreeableness-related adaptations
    if primary_spectrum == "agreeableness" or secondary_spectrum == "agreeableness":
        if primary["raw_intensity"] > 0.5 or (secondary_spectrum == "agreeableness" and secondary["raw_intensity"] > 0.5):
            adaptations.append(
                "When addressing conflicts or disagreements: Maintain your supportive approach while acknowledging different perspectives"
            )
        elif primary["raw_intensity"] < -0.5 or (secondary_spectrum == "agreeableness" and secondary["raw_intensity"] < -0.5):
            adaptations.append(
                "When providing feedback: Balance critical insights with constructive suggestions"
            )
    
    # Conscientiousness-related adaptations
    if primary_spectrum == "conscientiousness" or secondary_spectrum == "conscientiousness":
        if primary["raw_intensity"] > 0.5 or (secondary_spectrum == "conscientiousness" and secondary["raw_intensity"] > 0.5):
            adaptations.append(
                "When brainstorming creative ideas: Allow for some flexibility and exploration alongside your structured approach"
            )
        elif primary["raw_intensity"] < -0.5 or (secondary_spectrum == "conscientiousness" and secondary["raw_intensity"] < -0.5):
            adaptations.append(
                "When explaining procedures: Consider adding more structure to your naturally flexible style"
            )
    
    # Emotional Stability-related adaptations
    if primary_spectrum == "emotional_stability" or secondary_spectrum == "emotional_stability":
        if primary["raw_intensity"] < -0.5 or (secondary_spectrum == "emotional_stability" and secondary["raw_intensity"] < -0.5):
            adaptations.append(
                "When handling sensitive topics: Channel your emotional responsiveness into constructive empathy"
            )
        elif primary["raw_intensity"] > 0.5 or (secondary_spectrum == "emotional_stability" and secondary["raw_intensity"] > 0.5):
            adaptations.append(
                "When discussing emotional matters: Consider showing appropriate emotional acknowledgment alongside your calm demeanor"
            )
    
    # Openness-related adaptations
    if primary_spectrum == "openness" or secondary_spectrum == "openness":
        if primary["raw_intensity"] > 0.5 or (secondary_spectrum == "openness" and secondary["raw_intensity"] > 0.5):
            adaptations.append(
                "When providing practical guidance: Balance your creative approach with concrete, actionable suggestions"
            )
        elif primary["raw_intensity"] < -0.5 or (secondary_spectrum == "openness" and secondary["raw_intensity"] < -0.5):
            adaptations.append(
                "When exploring new ideas: Recognize the value of proven approaches while cautiously considering innovations"
            )
    
    # Format as bullet points
    return "\n".join(f"- {adaptation}" for adaptation in adaptations)


def generate_temperament_prompt(temperament_profile: TemperamentProfile) -> str:
    """
    Generate a system prompt incorporating the temperament profile.
    
    Args:
        temperament_profile: A TemperamentProfile object representing the personality state
        
    Returns:
        String containing the formatted system prompt
    """
    # Extract primary and secondary traits
    traits = extract_primary_secondary_traits(temperament_profile)
    
    # Format descriptions
    primary = format_trait_description(traits["primary"])
    secondary = format_trait_description(traits["secondary"])
    
    # Get linguistic features
    primary_features = get_linguistic_features(primary)
    secondary_features = get_linguistic_features(secondary)
    
    # Get adaptation rules
    adaptations = generate_adaptation_rules(primary, secondary)
    
    # Construct the prompt
    prompt = f"""You are an assistant with the following personality traits:
- Primary: {primary["direction"].capitalize()} {primary["spectrum"].replace("_", " ")}, expressing as {primary["trait"]} ({primary["intensity"]})
- Secondary: {secondary["direction"].capitalize()} {secondary["spectrum"].replace("_", " ")}, expressing as {secondary["trait"]} ({secondary["intensity"]})

Express these traits naturally through your language choices, tone, and expression style. 
For your primary trait, {primary_features}.
For your secondary trait, {secondary_features}.

Adapt your personality expression slightly based on context:
{adaptations}

Maintain consistency with your baseline personality profile while allowing for natural variation. 
Remember that these personality traits should influence how you communicate, not what information you provide.
"""
    
    return prompt


def create_temperament_profile_from_json(json_data: Dict[str, Any]) -> TemperamentProfile:
    """
    Create a TemperamentProfile from a JSON dictionary.
    
    Args:
        json_data: Dictionary containing trait spectrum values
        
    Returns:
        A TemperamentProfile object
    """
    profile = TemperamentProfile()
    
    # Map JSON keys to spectrum names
    spectrum_map = {
        "extraversion": TraitSpectrum.EXTRAVERSION,
        "agreeableness": TraitSpectrum.AGREEABLENESS,
        "conscientiousness": TraitSpectrum.CONSCIENTIOUSNESS,
        "emotional_stability": TraitSpectrum.EMOTIONAL_STABILITY,
        "openness": TraitSpectrum.OPENNESS
    }
    
    # Update each specified trait
    for spectrum_name, intensity in json_data.items():
        if spectrum_name in spectrum_map:
            setattr(profile, spectrum_name, TraitState(
                spectrum=spectrum_map[spectrum_name],
                intensity=float(intensity)
            ))
    
    return profile


def main():
    """Main function to run the script from command line."""
    parser = argparse.ArgumentParser(description="Generate temperament prompt for LLM agents based on Big Five model")
    parser.add_argument("--input", "-i", type=str, help="JSON file containing trait profile values")
    parser.add_argument("--output", "-o", type=str, help="Output file for the generated prompt")
    parser.add_argument("--extraversion", type=float, help="Intensity value for extraversion (-1.0 to 1.0)")
    parser.add_argument("--agreeableness", type=float, help="Intensity value for agreeableness (-1.0 to 1.0)")
    parser.add_argument("--conscientiousness", type=float, help="Intensity value for conscientiousness (-1.0 to 1.0)")
    parser.add_argument("--emotional-stability", type=float, help="Intensity value for emotional stability (-1.0 to 1.0)")
    parser.add_argument("--openness", type=float, help="Intensity value for openness (-1.0 to 1.0)")
    
    args = parser.parse_args()
    
    # Create temperament profile
    temperament_profile = None
    
    if args.input:
        # Load from JSON file
        try:
            with open(args.input, 'r') as f:
                json_data = json.load(f)
            temperament_profile = create_temperament_profile_from_json(json_data)
            logger.info(f"Loaded temperament profile from {args.input}")
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            return
    else:
        # Create from command line arguments
        profile_data = {}
        
        if args.extraversion is not None:
            profile_data["extraversion"] = args.extraversion
        if args.agreeableness is not None:
            profile_data["agreeableness"] = args.agreeableness
        if args.conscientiousness is not None:
            profile_data["conscientiousness"] = args.conscientiousness
        if args.emotional_stability is not None:
            profile_data["emotional_stability"] = args.emotional_stability
        if args.openness is not None:
            profile_data["openness"] = args.openness
        
        if profile_data:
            temperament_profile = create_temperament_profile_from_json(profile_data)
            logger.info("Created temperament profile from command line arguments")
        else:
            # Use default profile if no input specified
            temperament_profile = TemperamentProfile()
            logger.info("Using default temperament profile (balanced)")
    
    # Generate the prompt
    prompt = generate_temperament_prompt(temperament_profile)
    
    # Output prompt
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(prompt)
            logger.info(f"Wrote prompt to {args.output}")
        except Exception as e:
            logger.error(f"Error writing to output file: {e}")
            print(prompt)  # Print anyway if file writing fails
    else:
        print(prompt)


if __name__ == "__main__":
    main()