#!/usr/bin/env python3
"""
Gradio web application for the Temperament Agent chatbot.

This script creates a web interface using Gradio to interact with
the TemperamentAgent, displaying both the chat and personality traits.
"""

import os
import json
import time
import logging
import argparse
import numpy as np
import gradio as gr
import random
from typing import Dict, List, Any, Tuple
import dotenv
from openai import OpenAI

# Import our modules
from source.bigfivemodel import TemperamentProfile, TraitState, TraitSpectrum
from source.temperament_agent import TemperamentAgent

# Load environment variables from .env file
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("temperament_chat_app.log")
    ]
)
logger = logging.getLogger(__name__)


class TemperamentChatBot:
    """A chatbot that uses TemperamentAgent to manage its personality with weighted history."""
    
    def __init__(
        self,
        initial_profile: TemperamentProfile = None,
        model_name: str = "gpt-4.1-mini-2025-04-14",
        temperature: float = 0.7,
        max_tokens: int = 500,
        trait_decay_factor: float = 0.8,  # How much to weight older traits (0-1)
        max_trait_history: int = 5        # Maximum number of previous traits to consider
    ):
        """
        Initialize the TemperamentChatBot.
        
        Args:
            initial_profile: Initial personality profile
            model_name: OpenAI model to use (e.g., "gpt-4", "gpt-3.5-turbo")
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens in responses
            trait_decay_factor: How much to weight older traits (0-1)
            max_trait_history: Maximum number of previous traits to consider
        """
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Temperament memory parameters
        self.trait_decay_factor = max(0.0, min(1.0, trait_decay_factor))
        self.max_trait_history = max(1, max_trait_history)
        self.trait_history = []  # List of (timestamp, trait_dict) tuples
        
        # Create the temperament agent with a random or specified personality
        if initial_profile is None:
            initial_profile = self.generate_random_temperament()
        
        self.agent = TemperamentAgent(
            initial_profile=initial_profile,
            trait_adaptation_rate=0.2,
            base_system_prompt=(
                "You are a helpful AI assistant with a unique personality. "
                "Your goal is to provide accurate and useful information while connecting with the user."
            )
        )
        
        # Initialize conversation history
        self.messages = []
        
        logger.info(f"TemperamentChatBot initialized with model {model_name}")
    
    def generate_random_temperament(self) -> TemperamentProfile:
        """Generate a random temperament profile with varied traits."""
        # Generate random values between -1.0 and 1.0 for each trait
        profile = TemperamentProfile()
        
        # Set each trait to a random value (with a bias toward non-neutral values)
        for trait_name in ["extraversion", "agreeableness", "conscientiousness", "emotional_stability", "openness"]:
            # Use triangular distribution to bias toward more extreme values
            # (creates more interesting personalities than uniform distribution)
            value = random.triangular(-1.0, 1.0, 0.0)
            profile.adjust_trait(trait_name, value)
            
        logger.info(f"Generated random temperament profile")
        return profile
    
    def analyze_traits(self, text: str) -> Dict[str, float]:
        """
        Analyze traits in text using the OpenAI API.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary mapping Big Five trait names to intensity values
        """
        try:
            # Build a prompt for trait analysis
            analysis_prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are a personality analysis assistant. Analyze the text for personality traits "
                        "and provide a JSON object with trait intensities on the Big Five spectra:\n"
                        "- extraversion: -1.0 (extremely introverted) to 1.0 (extremely extraverted)\n"
                        "- agreeableness: -1.0 (highly disagreeable) to 1.0 (highly agreeable)\n"
                        "- conscientiousness: -1.0 (spontaneous/disorganized) to 1.0 (meticulous/organized)\n"
                        "- emotional_stability: -1.0 (anxious/volatile) to 1.0 (calm/stable)\n"
                        "- openness: -1.0 (conventional/practical) to 1.0 (creative/open to new ideas)\n"
                        "Return ONLY a valid JSON object with these fields and numerical values between -1.0 and 1.0."
                    )
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
            
            # Get trait analysis from the model
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=analysis_prompt,
                temperature=0.2,  # Lower temperature for more consistent analysis
                max_tokens=150
            )
            
            # Extract the JSON response
            analysis_text = response.choices[0].message.content.strip()
            
            # Find JSON object in the response
            start_idx = analysis_text.find('{')
            end_idx = analysis_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = analysis_text[start_idx:end_idx]
                traits = json.loads(json_str)
            else:
                logger.warning(f"Couldn't extract JSON from analysis: {analysis_text}")
                traits = {}
            
            # Ensure values are in the correct range
            for key, value in traits.items():
                traits[key] = max(-1.0, min(1.0, float(value)))
            
            # Add to trait history with current timestamp
            self.add_to_trait_history(traits)
            
            logger.info(f"Trait analysis: {traits}")
            return traits
            
        except Exception as e:
            logger.error(f"Error analyzing traits: {e}")
            return {}
    
    def add_to_trait_history(self, traits: Dict[str, float]) -> None:
        """
        Add traits to the history with current timestamp.
        
        Args:
            traits: Dictionary of trait intensities
        """
        # Add current traits to history with timestamp
        current_time = time.time()
        self.trait_history.append((current_time, traits))
        
        # Limit the history size
        if len(self.trait_history) > self.max_trait_history:
            self.trait_history = self.trait_history[-self.max_trait_history:]
    
    def calculate_weighted_traits(self) -> Dict[str, float]:
        """
        Calculate weighted traits based on history with decay.
        
        Returns:
            Dictionary of weighted trait intensities
        """
        if not self.trait_history:
            return {}
        
        # Initialize result with zeros
        all_trait_keys = set()
        for _, traits in self.trait_history:
            all_trait_keys.update(traits.keys())
        
        result = {key: 0.0 for key in all_trait_keys}
        
        # Calculate weights for each history entry
        total_weight = 0.0
        current_weight = 1.0  # Most recent has weight 1.0
        
        # Process from most recent to oldest
        for _, traits in reversed(self.trait_history):
            # Apply weights to each trait
            for key, value in traits.items():
                result[key] += value * current_weight
            
            # Track total weight applied
            total_weight += current_weight
            
            # Decay weight for older entries
            current_weight *= self.trait_decay_factor
        
        # Normalize by total weight
        if total_weight > 0:
            for key in result:
                result[key] /= total_weight
        
        return result
    
    def respond(self, user_message: str, chat_history: List = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response to the user message.
        
        Args:
            user_message: The user's message
            chat_history: Optional chat history from the UI
            
        Returns:
            Tuple of (assistant's response, trait state info)
        """
        try:
            # Analyze traits in the user message
            user_traits = self.analyze_traits(user_message)
            
            # Calculate weighted traits from history
            weighted_traits = self.calculate_weighted_traits()
            
            # Log the weighted traits
            logger.info(f"Weighted traits across conversation: {weighted_traits}")
            
            # Update agent's trait state with weighted traits
            # This gives a more stable response that considers conversation history
            self.agent.update_temperament(trait_adjustments=weighted_traits)
            
            # Get the current system prompt
            system_prompt = self.agent.get_system_prompt()
            
            # Prepare messages for API call, starting with the system prompt
            api_messages = [{"role": "system", "content": system_prompt}]
            
            # Add chat history if provided (from the UI)
            if chat_history:
                # Convert the UI history format to OpenAI API format
                for message in chat_history:
                    if isinstance(message, dict) and "role" in message and "content" in message:
                        api_messages.append(message)
            
            # Add the current user message
            api_messages.append({"role": "user", "content": user_message})
            
            # Update our internal history
            self.messages = api_messages.copy()
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=api_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract response text
            assistant_message = response.choices[0].message.content
            
            # Add assistant message to our internal history
            self.messages.append({"role": "assistant", "content": assistant_message})
            
            # Get trait state information for display
            trait_info = self._get_trait_display_info()
            
            # Record the interaction
            self.agent.record_interaction(
                user_message=user_message,
                agent_response=assistant_message,
                detected_user_traits=weighted_traits  # Use weighted traits
            )
            
            return assistant_message, trait_info
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an issue while processing your message. Could you try again?", {}
    
    def _get_trait_display_info(self) -> Dict[str, Any]:
        """
        Get formatted trait information for display.
        
        Returns:
            Dictionary with trait information
        """
        # Get dominant traits
        dominant_traits = self.agent.profile.get_dominant_traits(n=3)
        
        # Get all trait values as a dictionary
        trait_values = {
            "extraversion": self.agent.profile.extraversion.intensity,
            "agreeableness": self.agent.profile.agreeableness.intensity,
            "conscientiousness": self.agent.profile.conscientiousness.intensity,
            "emotional_stability": self.agent.profile.emotional_stability.intensity,
            "openness": self.agent.profile.openness.intensity
        }
        
        # Format the dominant traits for display
        dominant_formatted = []
        for trait in dominant_traits:
            formatted = f"{trait['trait']} ({trait['spectrum']}: {trait['intensity']:.2f})"
            dominant_formatted.append(formatted)
        
        return {
            "dominant_traits": dominant_formatted,
            "trait_values": trait_values
        }
    
    def reset_conversation(self) -> None:
        """Reset the conversation history while maintaining personality state."""
        self.messages = []
        logger.info("Reset conversation history")


def format_message_for_display(message):
    """Format a message for display, extracting just the text if it contains trait data."""
    if isinstance(message, tuple) and len(message) > 0:
        return message[0]  # Extract just the text part from (text, trait_info) tuple
    return message


def create_trait_chart(trait_values):
    """Create a Big Five trait chart using Plotly."""
    import plotly.graph_objects as go
    from source.bigfivemodel import TRAIT_MAPPINGS, TraitSpectrum
    
    if not trait_values:
        # Return empty chart if no values
        fig = go.Figure()
        fig.add_annotation(
            text="No personality trait data available",
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(title="Big Five Personality Profile")
        return fig
    
    # Define the intensity values
    intensities = [-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0]
    intensity_labels = [str(i) for i in intensities]
    
    # Define Big Five spectrum colors
    spectrum_colors = {
        "extraversion": "rgba(255, 100, 100, 0.8)",     # Red
        "agreeableness": "rgba(100, 200, 100, 0.8)",    # Green
        "conscientiousness": "rgba(100, 100, 255, 0.8)", # Blue
        "emotional_stability": "rgba(255, 200, 100, 0.8)", # Orange
        "openness": "rgba(200, 100, 200, 0.8)"          # Purple
    }
    
    # Create a figure
    fig = go.Figure()
    
    # Define the order of spectra
    spectra_order = [
        "extraversion", 
        "agreeableness", 
        "conscientiousness", 
        "emotional_stability", 
        "openness"
    ]
    
    # Add horizontal bars for each spectrum with markers at current value
    y_positions = []
    current_y = 5
    
    for spectrum_name in spectra_order:
        current_y -= 1
        y_positions.append(current_y)
        
        # Get the current value for this spectrum
        current_value = trait_values.get(spectrum_name, 0)
        
        # Add a line representing the spectrum
        fig.add_trace(go.Scatter(
            x=intensities,
            y=[current_y] * len(intensities),
            mode='lines',
            line=dict(color=spectrum_colors[spectrum_name], width=12),
            name=spectrum_name.replace('_', ' ').title(),
            showlegend=False
        ))
        
        # Add marker for current value
        fig.add_trace(go.Scatter(
            x=[current_value],
            y=[current_y],
            mode='markers',
            marker=dict(color='black', size=14, line=dict(color='white', width=2)),
            name=f"Current: {spectrum_name.replace('_', ' ').title()}",
            showlegend=False
        ))
        
        # Get the current trait name for this spectrum
        current_intensity_levels = list(TRAIT_MAPPINGS[getattr(TraitSpectrum, spectrum_name.upper())].keys())
        closest_intensity = min(current_intensity_levels, key=lambda x: abs(x - current_value))
        current_trait = TRAIT_MAPPINGS[getattr(TraitSpectrum, spectrum_name.upper())][closest_intensity]
        
        # Add trait words for each position
        for i, intensity in enumerate(intensities):
            trait_name = TRAIT_MAPPINGS[getattr(TraitSpectrum, spectrum_name.upper())][intensity]
            
            # Determine text style
            text_color = 'black'
            text_size = 12
            
            fig.add_annotation(
                x=intensity,
                y=current_y + 0.2,  # Position above the line
                text=trait_name,
                showarrow=False,
                font=dict(color=text_color, size=text_size, family="Arial"),
                xanchor='center',
                yanchor='bottom'
            )
        
        # Add spectrum name
        formatted_name = spectrum_name.replace("_", " ").title()
        fig.add_annotation(
            x=-1.2,
            y=current_y,
            text=formatted_name,
            showarrow=False,
            font=dict(color='black', size=14, family="Arial"),
            xanchor='right',
            yanchor='middle'
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Big Five Personality Profile",
            'font': {'size': 22}
        },
        xaxis=dict(
            title="Trait Intensity",
            tickvals=intensities,
            ticktext=intensity_labels,
            range=[-1.3, 1.1],
        ),
        yaxis=dict(
            showticklabels=False,
            range=[-0.5, 5.5]
        ),
        height=600,
        margin=dict(l=120, r=20, t=60, b=50),
        plot_bgcolor='white',
        showlegend=False
    )
    
    # Add gray gridlines at each intensity value
    for intensity in intensities:
        fig.add_shape(
            type="line",
            x0=intensity,
            x1=intensity,
            y0=-0.5,
            y1=5.5,
            line=dict(color="rgba(200,200,200,0.5)", width=1, dash="dot")
        )
    
    return fig


def create_process_message_handler(bot):
    """Create a message processing function with a reference to the bot."""
    def process_message(user_message, history):
        """Process a user message and update the UI."""
        # Check empty input
        if not user_message.strip():
            return "", history, None, ""
        
        # Convert history to a list if it's not already (for safety)
        history_list = list(history) if history else []
        
        # Get response and trait state - pass the chat history
        response, trait_info = bot.respond(user_message, history_list)
        
        # Generate chart from trait values
        trait_chart = create_trait_chart(trait_info.get("trait_values", {}))
        
        # Format dominant traits text
        dominant_traits_text = "Dominant traits: "
        if "dominant_traits" in trait_info and trait_info["dominant_traits"]:
            dominant_traits_text += ", ".join(trait_info["dominant_traits"])
        else:
            dominant_traits_text += "None"
        
        # Format the response - just get the text part if it's a tuple
        response_text = format_message_for_display(response)
        
        # Add new messages to history
        # Make a copy of the history to avoid modifying the original
        updated_history = list(history) if history else []
        updated_history.append({"role": "user", "content": user_message})
        updated_history.append({"role": "assistant", "content": response_text})
        
        # Return the necessary information
        return "", updated_history, trait_chart, dominant_traits_text
    
    return process_message


def create_reset_chat_handler(bot):
    """Create a reset function with a reference to the bot."""
    def reset_chat(history):
        """Reset the chat history and return a clean state."""
        import plotly.graph_objects as go
        
        bot.reset_conversation()
        
        # Create empty figure for the reset state
        fig = go.Figure()
        fig.add_annotation(
            text="No personality trait data available",
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(title="Big Five Personality Profile")
        
        # Return empty list for the chat history
        empty_history = []
        return empty_history, fig, "Dominant traits: None"
    
    return reset_chat


def create_ui(bot):
    """Create the Gradio UI for the chatbot."""
    with gr.Blocks(title="Temperament AI Chatbot") as app:
        gr.Markdown("# Temperament AI Chatbot")
        gr.Markdown("This chatbot uses the Big Five personality model to simulate a unique temperament in an AI assistant.")
        
        # Create handlers with access to the bot
        process_message_fn = create_process_message_handler(bot)
        reset_chat_fn = create_reset_chat_handler(bot)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    show_copy_button=True,
                    type="messages"
                )
                
                # Message input
                msg = gr.Textbox(
                    label="Your message",
                    placeholder="Type your message here...",
                    lines=3,
                    max_lines=10
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Reset Chat", variant="secondary")
            
            with gr.Column(scale=1):
                # Trait visualization
                gr.Markdown("## Bot's Personality Profile")
                # Use gr.Plot which works with Plotly figures
                trait_chart = gr.Plot(label="Big Five Personality Profile")
                dominant_traits = gr.Markdown("Dominant traits: None")
        
        # Event handlers
        submit_btn.click(
            fn=process_message_fn,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, trait_chart, dominant_traits]
        )
        
        msg.submit(
            fn=process_message_fn,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, trait_chart, dominant_traits]
        )
        
        clear_btn.click(
            fn=reset_chat_fn,
            inputs=[chatbot],
            outputs=[chatbot, trait_chart, dominant_traits]
        )
    
    return app


def main():
    """Run the Gradio application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Temperament chatbot with Gradio UI")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini-2025-04-14", help="OpenAI model to use")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--share", action="store_true", help="Create a shareable link")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="The IP address to bind the server to")
    parser.add_argument("--profile", type=str, help="JSON file with personality profile (if not provided, uses random)")
    
    args = parser.parse_args()
    
    # Load profile if specified
    initial_profile = None
    if args.profile:
        try:
            with open(args.profile, 'r') as f:
                json_data = json.load(f)
                
            from source.temperament_prompt_generator import create_temperament_profile_from_json
            initial_profile = create_temperament_profile_from_json(json_data)
            logger.info(f"Loaded personality profile from {args.profile}")
        except Exception as e:
            logger.error(f"Error loading profile: {e}")
    
    # Create the chatbot
    bot = TemperamentChatBot(
        initial_profile=initial_profile,
        model_name=args.model,
        temperature=args.temp
    )
    
    # Create and launch the UI
    app = create_ui(bot)
    app.launch(server_name=args.server_name, share=args.share)


if __name__ == "__main__":
    main()