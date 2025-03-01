"""
Centralized prompt system for the Zettelkasten project.

This package provides a unified interface for accessing and managing prompt templates
across different models and use cases in the application.

The main components are:
- PromptRegistry: Central registry for all prompt templates
- PromptTemplate: Base class for prompt templates with metadata
- template_registry: Global singleton instance of the registry

Example usage:
    from core.prompts import template_registry
    
    # Get a prompt template by name
    template = template_registry.get("first_pass")
    
    # Format a prompt with parameters
    prompt = template.format(window_text="...")
    
    # Register a custom prompt
    template_registry.register("custom_prompt", 
                              "My custom prompt template with {param}",
                              required_params=["param"])
"""

from core.prompts.registry import PromptRegistry, template_registry
from core.prompts.template import PromptTemplate

__all__ = ['PromptRegistry', 'PromptTemplate', 'template_registry']