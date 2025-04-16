# Temperament Modeling for LLM Agents

![Temperament Bot](emotionalbot.png)

This repository implements a Linguistic Personality Adaptation Framework for LLM Agents. It provides a complete framework for modeling, managing, and integrating personality traits into LLM-powered agents based on the Big Five personality model.

## Overview

The Big Five Linguistic Adaptation Framework offers a structured approach to modeling personality in LLMs through five distinct trait spectra, each with varying intensities from negative to positive. This implementation provides:

1. A complete Pydantic v2 model of the Big Five personality traits
2. Tools for generating personality-aware system prompts
3. A `TemperamentAgent` class for managing personality states
4. Example integrations with LLM APIs

## Components

### 1. Temperament Model (`source/beammodel.py`)

The core implementation of the Big Five personality model as a Pydantic model, including:

- Five trait spectra (extraversion, agreeableness, conscientiousness, emotional stability, openness)
- Quantified intensity levels (-1.0 to 1.0)
- Trait state representation and manipulation
- Methods for blending personality profiles

### 2. Temperament Prompt Generator (`source/emotional_prompt_generator.py`)

Transforms personality traits into natural language system prompts by:

- Extracting primary and secondary traits from profiles
- Formatting appropriate linguistic expressions
- Generating contextual adaptation rules
- Producing complete system prompts

### 3. Temperament Agent (`source/emotional_agent.py`)

A class for managing LLM personality states with features for:

- Dynamic trait state updates
- Adaptation to detected user traits
- Interaction recording and state persistence

### 4. LLM Integration (`integration_with_llm.py`)

Example of integrating the temperament framework with OpenAI's API to create an interactive personality-aware chatbot.

## Usage

### Basic Usage

```python
# Import the components
from source.beammodel import TemperamentProfile, TraitState, TraitSpectrum
from source.emotional_prompt_generator import generate_temperament_prompt

# Create a personality profile
profile = TemperamentProfile(
    extraversion=TraitState(spectrum=TraitSpectrum.EXTRAVERSION, intensity=0.6),  # Exuberant
    agreeableness=TraitState(spectrum=TraitSpectrum.AGREEABLENESS, intensity=0.3)  # Cooperative
)

# Generate a system prompt with personality guidelines
prompt = generate_temperament_prompt(profile)
print(prompt)
```

### Using the Temperament Agent

```python
from source.beammodel import TemperamentProfile
from source.emotional_agent import TemperamentAgent

# Create an agent with an initial profile
agent = TemperamentAgent(
    initial_profile=profile,
    trait_adaptation_rate=0.2,
    base_system_prompt="You are a helpful assistant."
)

# Update the trait state based on user input
agent.update_temperament(
    trait_adjustments={"openness": 0.3}  # Increase openness
)

# Get the current system prompt
system_prompt = agent.get_system_prompt()
```

### Interactive Chat

Run the interactive chat example:

```bash
# Make sure you've set up your .env file with your OpenAI API key
# Then run the interactive chat
python integration_with_llm.py
```

## Installation

### Standard Installation

1. Clone the repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Copy `example.env` to `.env`
   - Replace the placeholder API key with your actual OpenAI API key
   ```bash
   cp example.env .env
   # Edit .env file with your API key
   ```

### Docker Installation

You can also run the application using Docker:

1. Clone the repository
2. Create your .env file with your OpenAI API key:
   ```bash
   cp example.env .env
   # Edit .env file with your API key
   ```
3. Build and run with docker-compose:
   ```bash
   docker-compose up -d
   ```
4. Access the application at http://localhost:7860 or http://[your-machine-ip]:7860 from other devices on your network

To stop the application:
```bash
docker-compose down
```

## Running Tests

The project uses pytest for testing. To run the tests:

```bash
python -m pytest
```

For more verbose output:

```bash
python -m pytest -v
```

## Examples

The repository includes several examples:

- `emotional_prompt_generator_demo.py`: Demonstrates creating personality profiles and generating prompts
- `integration_with_llm.py`: Shows how to create an interactive command-line chatbot with personality awareness
- `emotional_chatbot_app.py`: Provides a web interface for the temperament chatbot using Gradio

### Running the Web Interface

To run the Gradio web interface:

```bash
# Make sure you've set up your .env file with your OpenAI API key
python emotional_chatbot_app.py
```

To make the interface accessible from other devices on your network:

```bash
python emotional_chatbot_app.py --server-name 0.0.0.0
```

For additional options:

```bash
python emotional_chatbot_app.py --help
```

Command line options:
- `--model`: Specify the OpenAI model to use (default: gpt-4.1-mini-2025-04-14)
- `--temp`: Set the temperature for generation (default: 0.7)
- `--server-name`: Host to bind the server to (use 0.0.0.0 to make it accessible on your network)
- `--share`: Create a shareable link to access the app from another device
- `--profile`: Path to a JSON file with a pre-defined personality profile (if not provided, uses random)

The web interface displays:
- A chat interface for conversing with the AI
- A graph showing the bot's personality profile across all trait spectra
- A summary of the dominant traits in the agent's personality

#### Random Personality Generation

The chatbot application can generate random personality profiles for unique and diverse agents:
- Randomly assigns intensity values to each of the Big Five traits
- Creates more interesting personalities by biasing toward non-neutral values
- Allows testing different personality combinations without manual configuration

## License

MIT

## Acknowledgments

Inspired by research in personality psychology and the Big Five personality model as a framework for computational modeling of personality in AI systems.