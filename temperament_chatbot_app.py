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
    """A chatbot that uses TemperamentAgent to manage its personality with user-adjustable traits."""
    
    def __init__(
        self,
        initial_profile: TemperamentProfile = None,
        model_name: str = "gpt-4.1-mini-2025-04-14",
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """
        Initialize the TemperamentChatBot.
        
        Args:
            initial_profile: Initial personality profile
            model_name: OpenAI model to use (e.g., "gpt-4", "gpt-3.5-turbo")
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens in responses
        """
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Create the temperament agent with a random or specified personality
        if initial_profile is None:
            initial_profile = self.generate_random_temperament()
        
        self.agent = TemperamentAgent(
            initial_profile=initial_profile,
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
    
    # No longer need trait analysis functionality since we're using sliders
    
    def respond(self, user_message: str, chat_history: List = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response to the user message.
        
        Args:
            user_message: The user's message
            chat_history: Optional chat history from the UI
            
        Returns:
            Tuple of (assistant's response, empty dict for backward compatibility)
        """
        try:
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
            
            for api_message in api_messages:
                # Log the message for debugging
                print(f"{api_message['role'].capitalize()}: {api_message['content']}")

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
            
            # Record the interaction
            self.agent.record_interaction(
                user_message=user_message,
                agent_response=assistant_message
            )
            
            return assistant_message, {}
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an issue while processing your message. Could you try again?", {}
    
    # No longer need the trait display info method since we're using sliders
    
    def reset_conversation(self) -> None:
        """Reset the conversation history while maintaining personality state."""
        self.messages = []
        logger.info("Reset conversation history")


def format_message_for_display(message):
    """Format a message for display, extracting just the text if it contains trait data."""
    if isinstance(message, tuple) and len(message) > 0:
        return message[0]  # Extract just the text part from (text, trait_info) tuple
    return message


# Removed Plotly chart creation function as it's no longer needed


def create_process_message_handler(bot):
    """Create a message processing function with a reference to the bot."""
    def process_message(user_message, history):
        """Process a user message and update the UI."""
        # Check empty input
        if not user_message.strip():
            return "", history, ""
        
        # Convert history to a list if it's not already (for safety)
        history_list = list(history) if history else []
        
        # Get response - pass the chat history
        # Note: We no longer need trait_info since we're using sliders
        response, _ = bot.respond(user_message, history_list)
        
        # Format the response - just get the text part if it's a tuple
        response_text = format_message_for_display(response)
        
        # Add new messages to history
        # Make a copy of the history to avoid modifying the original
        updated_history = list(history) if history else []
        updated_history.append({"role": "user", "content": user_message})
        updated_history.append({"role": "assistant", "content": response_text})
        
        # Return the necessary information
        return "", updated_history, ""
    
    return process_message


def create_reset_chat_handler(bot):
    """Create a reset function with a reference to the bot."""
    def reset_chat(history):
        """Reset the chat history and return a clean state."""
        bot.reset_conversation()
        
        # Return empty list for the chat history
        empty_history = []
        return empty_history, ""
    
    return reset_chat


def create_ui(bot):
    """Create the Gradio UI for the chatbot."""
    with gr.Blocks(title="Temperament AI Chatbot") as app:
        gr.Markdown("# Temperament AI Chatbot")
        gr.Markdown("This chatbot uses the Big Five personality model with adjustable temperament traits.")
        
        # Create handlers with access to the bot
        process_message_fn = create_process_message_handler(bot)
        reset_chat_fn = create_reset_chat_handler(bot)
        
        # Function to update a single trait
        def update_trait(trait_name, value):
            # Update the specific trait
            from source.bigfivemodel import TraitState, TraitSpectrum
            spectrum = getattr(TraitSpectrum, trait_name.upper())
            trait_state = TraitState(spectrum=spectrum, intensity=value)
            
            # Set the trait in the agent's profile
            setattr(bot.agent.profile, trait_name, trait_state)
            
            # Update the system prompt
            bot.agent.current_prompt = bot.agent._generate_prompt()
            
            # Return the trait name
            return get_trait_name_for_value(trait_name, value)
        
        # Function to get trait name for slider value
        def get_trait_name_for_value(trait_name, value):
            from source.bigfivemodel import TRAIT_MAPPINGS, TraitSpectrum
            
            # Get the trait spectrum
            spectrum = getattr(TraitSpectrum, trait_name.upper())
            
            # Find the closest intensity level
            intensity_levels = list(TRAIT_MAPPINGS[spectrum].keys())
            closest_intensity = min(intensity_levels, key=lambda x: abs(x - value))
            
            # Return the trait name
            return TRAIT_MAPPINGS[spectrum][closest_intensity]
        
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
                # Temperament controls
                gr.Markdown("## Adjust Bot's Personality")
                
                # Get initial values from bot's profile
                initial_extraversion = bot.agent.profile.extraversion.intensity
                initial_agreeableness = bot.agent.profile.agreeableness.intensity
                initial_conscientiousness = bot.agent.profile.conscientiousness.intensity
                initial_emotional_stability = bot.agent.profile.emotional_stability.intensity
                initial_openness = bot.agent.profile.openness.intensity
                
                with gr.Row():
                    extraversion_slider = gr.Slider(
                        minimum=-1.0, maximum=1.0, step=0.1, 
                        value=initial_extraversion,
                        label="Extraversion"
                    )
                    extraversion_trait = gr.Textbox(
                        value=get_trait_name_for_value("extraversion", initial_extraversion),
                        label="Trait",
                        interactive=False
                    )
                
                with gr.Row():
                    agreeableness_slider = gr.Slider(
                        minimum=-1.0, maximum=1.0, step=0.1, 
                        value=initial_agreeableness,
                        label="Agreeableness"
                    )
                    agreeableness_trait = gr.Textbox(
                        value=get_trait_name_for_value("agreeableness", initial_agreeableness),
                        label="Trait",
                        interactive=False
                    )
                
                with gr.Row():
                    conscientiousness_slider = gr.Slider(
                        minimum=-1.0, maximum=1.0, step=0.1, 
                        value=initial_conscientiousness,
                        label="Conscientiousness"
                    )
                    conscientiousness_trait = gr.Textbox(
                        value=get_trait_name_for_value("conscientiousness", initial_conscientiousness),
                        label="Trait",
                        interactive=False
                    )
                
                with gr.Row():
                    emotional_stability_slider = gr.Slider(
                        minimum=-1.0, maximum=1.0, step=0.1, 
                        value=initial_emotional_stability,
                        label="Emotional Stability"
                    )
                    emotional_stability_trait = gr.Textbox(
                        value=get_trait_name_for_value("emotional_stability", initial_emotional_stability),
                        label="Trait",
                        interactive=False
                    )
                
                with gr.Row():
                    openness_slider = gr.Slider(
                        minimum=-1.0, maximum=1.0, step=0.1, 
                        value=initial_openness,
                        label="Openness"
                    )
                    openness_trait = gr.Textbox(
                        value=get_trait_name_for_value("openness", initial_openness),
                        label="Trait",
                        interactive=False
                    )
                
                # Status message
                status_msg = gr.Markdown("Adjust the sliders to change the bot's personality traits")
        
        # Event handlers for chat
        submit_btn.click(
            fn=process_message_fn,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, status_msg]
        )
        
        msg.submit(
            fn=process_message_fn,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, status_msg]
        )
        
        clear_btn.click(
            fn=reset_chat_fn,
            inputs=[chatbot],
            outputs=[chatbot, status_msg]
        )
        
        # Individual slider change events to update trait text and bot's personality instantly
        extraversion_slider.change(
            fn=lambda x: update_trait("extraversion", x),
            inputs=[extraversion_slider],
            outputs=[extraversion_trait]
        )
        
        agreeableness_slider.change(
            fn=lambda x: update_trait("agreeableness", x),
            inputs=[agreeableness_slider],
            outputs=[agreeableness_trait]
        )
        
        conscientiousness_slider.change(
            fn=lambda x: update_trait("conscientiousness", x),
            inputs=[conscientiousness_slider],
            outputs=[conscientiousness_trait]
        )
        
        emotional_stability_slider.change(
            fn=lambda x: update_trait("emotional_stability", x),
            inputs=[emotional_stability_slider],
            outputs=[emotional_stability_trait]
        )
        
        openness_slider.change(
            fn=lambda x: update_trait("openness", x),
            inputs=[openness_slider],
            outputs=[openness_trait]
        )
    
    return app


def main():
    """Run the Gradio application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Temperament chatbot with adjustable Big Five traits")
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
    app.launch(server_name=args.server_name, share=args.share, 
               height=900)  # Increased height to accommodate sliders


if __name__ == "__main__":
    main()