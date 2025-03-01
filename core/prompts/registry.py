"""
PromptRegistry class for the centralized prompt system.

This module provides the central registry for all prompt templates used in the application.
"""
from typing import Dict, List, Optional, Any, Union
import importlib
import yaml
import os
import logging
import inspect
from pathlib import Path

from core.prompts.template import PromptTemplate
from utils.logging import get_logger

logger = get_logger(__name__)


class PromptRegistry:
    """
    Central registry for prompt templates.
    
    This class provides a unified interface for accessing prompt templates
    across different models and use cases. It handles template loading,
    registration, and retrieval, with support for model-specific overrides.
    """
    
    def __init__(self):
        """Initialize an empty registry."""
        self._templates: Dict[str, Dict[str, PromptTemplate]] = {
            "default": {}  # Default templates (not model-specific)
        }
        self._initialized = False
    
    def register(
        self, 
        name: str, 
        template: Union[str, PromptTemplate], 
        model: Optional[str] = None,
        description: Optional[str] = None,
        required_params: Optional[List[str]] = None,
        version: str = "1.0",
        tags: Optional[List[str]] = None,
        override: bool = False
    ) -> PromptTemplate:
        """
        Register a prompt template.
        
        Args:
            name: Unique identifier for the template
            template: The template string or PromptTemplate object
            model: Optional model name for model-specific templates
            description: Human-readable description of the template's purpose
            required_params: List of parameters that must be provided when formatting
            version: Version of the template for tracking changes
            tags: List of tags for categorizing templates
            override: Whether to override an existing template with the same name/model
            
        Returns:
            The registered PromptTemplate object
            
        Raises:
            ValueError: If a template with the same name/model already exists and override=False
        """
        # Get the appropriate registry based on model
        model_registry = self._get_model_registry(model)
        
        # Check if template already exists
        if name in model_registry and not override:
            raise ValueError(f"Template '{name}' for model '{model or 'default'}' already exists. Use override=True to replace it.")
        
        # Convert string to PromptTemplate if needed
        if isinstance(template, str):
            template_obj = PromptTemplate(
                name=name,
                template=template,
                description=description,
                required_params=required_params,
                model=model,
                version=version,
                tags=tags
            )
        else:
            template_obj = template
        
        # Register the template
        model_registry[name] = template_obj
        logger.debug(f"Registered template '{name}' for model '{model or 'default'}'")
        
        return template_obj
    
    def get(self, name: str, model: Optional[str] = None) -> PromptTemplate:
        """
        Get a prompt template by name.
        
        Args:
            name: Name of the template to retrieve
            model: Model to retrieve the template for (uses default if not found)
            
        Returns:
            The requested PromptTemplate object
            
        Raises:
            KeyError: If no template with the given name exists
        """
        # Try to get model-specific template first
        if model and model in self._templates and name in self._templates[model]:
            return self._templates[model][name]
        
        # Fall back to default template
        if name in self._templates["default"]:
            return self._templates["default"][name]
        
        raise KeyError(f"No template found with name '{name}' for model '{model or 'default'}'")
    
    def has_template(self, name: str, model: Optional[str] = None) -> bool:
        """Check if a template exists."""
        # Check model-specific template first
        if model and model in self._templates and name in self._templates[model]:
            return True
        
        # Check default template
        return name in self._templates["default"]
    
    def format(self, name: str, model: Optional[str] = None, **kwargs) -> str:
        """
        Format a template with parameters.
        
        Args:
            name: Name of the template to format
            model: Model to retrieve the template for
            **kwargs: Parameters to fill in the template
            
        Returns:
            The formatted prompt string
        """
        template = self.get(name, model)
        return template.format(**kwargs)
    
    def _get_model_registry(self, model: Optional[str] = None) -> Dict[str, PromptTemplate]:
        """Get the registry for a specific model, creating it if needed."""
        model_key = model or "default"
        if model_key not in self._templates:
            self._templates[model_key] = {}
        return self._templates[model_key]
    
    def list_templates(self, model: Optional[str] = None) -> List[str]:
        """List all template names for a given model (or default)."""
        if model and model in self._templates:
            return list(self._templates[model].keys())
        return list(self._templates["default"].keys())
    
    def list_all_templates(self) -> Dict[str, List[str]]:
        """List all templates grouped by model."""
        return {model: list(templates.keys()) for model, templates in self._templates.items()}
    
    def load_from_module(self, module_path: str, model: Optional[str] = None) -> None:
        """
        Load templates from a Python module.
        
        Args:
            module_path: Dotted path to the module (e.g., 'config.chunking.prompts')
            model: Model to associate with these templates
        """
        try:
            # Try to import the module
            module = importlib.import_module(module_path)
            
            # Get all uppercase variables as templates
            for name, value in inspect.getmembers(module):
                if name.isupper() and isinstance(value, str):
                    # Register the template
                    self.register(
                        name=name.lower(),  # Convert to lowercase for consistency
                        template=value,
                        model=model,
                        description=f"Imported from {module_path}.{name}",
                        override=True  # Override existing templates
                    )
            
            logger.info(f"Loaded templates from module {module_path}")
        except (ImportError, AttributeError) as e:
            logger.warning(f"Failed to load templates from module {module_path}: {str(e)}")
    
    def load_from_yaml(self, yaml_path: str, model: Optional[str] = None) -> None:
        """
        Load templates from a YAML file.
        
        Args:
            yaml_path: Path to the YAML file
            model: Model to associate with these templates
        """
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Check if the file has a prompts section
            templates_data = data.get('prompts', data)
            
            # Register each template
            for name, template_data in templates_data.items():
                if isinstance(template_data, str):
                    # Simple string template
                    self.register(
                        name=name,
                        template=template_data,
                        model=model,
                        override=True
                    )
                elif isinstance(template_data, dict):
                    # Template with metadata
                    self.register(
                        name=name,
                        template=template_data['template'],
                        model=model or template_data.get('model'),
                        description=template_data.get('description'),
                        required_params=template_data.get('required_params'),
                        version=template_data.get('version', '1.0'),
                        tags=template_data.get('tags'),
                        override=True
                    )
            
            logger.info(f"Loaded templates from YAML file {yaml_path}")
        except (IOError, yaml.YAMLError) as e:
            logger.warning(f"Failed to load templates from YAML file {yaml_path}: {str(e)}")
    
    def load_defaults(self) -> None:
        """
        Load default templates from bundled modules.
        
        This method loads the standard templates shipped with the package.
        """
        if self._initialized:
            return
        
        try:
            # First try to load from the new centralized defaults
            from core.prompts.defaults import DEFAULT_PROMPTS, MODEL_SPECIFIC_PROMPTS
            
            # Register default prompts
            for name, template_data in DEFAULT_PROMPTS.items():
                self.register(
                    name=name,
                    template=template_data["template"],
                    description=template_data.get("description"),
                    required_params=template_data.get("required_params"),
                    version=template_data.get("version", "1.0"),
                    tags=template_data.get("tags"),
                    override=True
                )
            
            # Register model-specific prompts
            for model, prompts in MODEL_SPECIFIC_PROMPTS.items():
                for name, template_data in prompts.items():
                    self.register(
                        name=name,
                        template=template_data["template"],
                        model=model,
                        description=template_data.get("description"),
                        required_params=template_data.get("required_params"),
                        version=template_data.get("version", "1.0"),
                        tags=template_data.get("tags"),
                        override=True
                    )
            
            logger.info("Loaded default prompt templates from centralized system")
        except ImportError:
            # Fall back to legacy prompt loading
            logger.warning("Could not load from centralized defaults, falling back to legacy mode")
            
            # Load from standard prompt modules
            self.load_from_module('config.prompts')
            self.load_from_module('config.chunking.prompts')
            
            # Load model-specific prompts
            self.load_from_module('config.chunking.prompts_mistral', model='mistral')
            
            logger.info("Loaded default prompt templates from legacy modules")
        
        self._initialized = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to a dictionary for serialization."""
        result = {}
        for model, templates in self._templates.items():
            result[model] = {name: template.to_dict() for name, template in templates.items()}
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptRegistry':
        """Create a registry from a dictionary."""
        registry = cls()
        for model, templates in data.items():
            for name, template_data in templates.items():
                template = PromptTemplate.from_dict(template_data)
                registry.register(name, template, model=model, override=True)
        return registry


# Singleton instance for global access
template_registry = PromptRegistry()
template_registry.load_defaults()