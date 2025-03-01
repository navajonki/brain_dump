from typing import Dict, Any, Optional, Union
import logging
from core.prompts import template_registry, PromptTemplate

class ChunkingConfig:
    """Configuration for the chunking process."""
    
    def __init__(self, **kwargs):
        # Model configuration
        self.model = kwargs.get('model', 'gpt-3.5-turbo')
        self.llm_backend = kwargs.get('llm_backend', 'openai')
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens', 4000)
        self.max_response_tokens = kwargs.get('max_response_tokens', 1000)
        self.ollama_url = kwargs.get('ollama_url', 'http://localhost:11434/api/generate')
        
        # Window configuration
        self.window_size = kwargs.get('window_size', 1000)
        self.overlap_size = kwargs.get('overlap_size', 100)
        
        # Feature flags
        self.use_function_calling = kwargs.get('use_function_calling', False)
        self.validate_function_output = kwargs.get('validate_function_output', False)
        self.global_check_enabled = kwargs.get('global_check_enabled', True)
        self.tagging_enabled = kwargs.get('tagging_enabled', True)
        self.relationships_enabled = kwargs.get('relationships_enabled', True)
        self.track_transcript_positions = kwargs.get('track_transcript_positions', True)
        
        # Initialize the prompt registry
        self.prompt_registry = template_registry
        
        # Create legacy prompt access objects for backward compatibility
        self.prompts = type('Prompts', (), {})()
        
        # Load custom prompts from YAML config
        self._load_prompt_overrides(kwargs)
        
        # Function schemas
        self.function_schemas = kwargs.get('function_schemas', {})
        
        # Output settings
        self.output_dir = kwargs.get('output_dir', "output/chunks")
        self.debug = kwargs.get('debug', True)
        
        # Load custom prompts if a different module is specified
        if 'prompt_template_module' in kwargs:
            self.prompt_template_module = kwargs.get('prompt_template_module')
            self._load_prompts_from_module()
    
    def _load_prompt_overrides(self, kwargs: Dict[str, Any]) -> None:
        """Load prompt overrides from config kwargs."""
        # Define the mapping of prompt names
        prompt_keys = [
            'first_pass_prompt', 
            'second_pass_prompt', 
            'global_check_prompt',
            'tagging_prompt',
            'relationship_prompt'
        ]
        
        # Check for and register prompt overrides from config
        for key in prompt_keys:
            if key in kwargs and kwargs[key]:
                # Get normalized name (remove '_prompt' suffix)
                normalized_name = key.replace('_prompt', '')
                
                # Register the override in the prompt registry
                self.prompt_registry.register(
                    name=normalized_name,
                    template=kwargs[key],
                    model=self.model if self.use_function_calling else None,
                    description=f"Custom {normalized_name} prompt from config",
                    override=True
                )
                
                # Store for direct attribute access (legacy style)
                setattr(self, key, kwargs[key])
                
                # Also set uppercase version in prompts namespace (very legacy style)
                uppercase_key = key.upper()
                setattr(self.prompts, uppercase_key, kwargs[key])
    
    def _load_prompts_from_module(self) -> None:
        """Load prompt templates from the configured module."""
        try:
            if not hasattr(self, 'prompt_template_module') or not self.prompt_template_module:
                return
            
            # Use the prompt registry to load from module
            self.prompt_registry.load_from_module(
                self.prompt_template_module,
                model=self.model if self.use_function_calling else None
            )
            
            # Update legacy access attributes
            self._update_legacy_prompt_attributes()
            
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Failed to load prompt templates from {self.prompt_template_module}: {str(e)}"
            )
    
    def _update_legacy_prompt_attributes(self) -> None:
        """Update legacy prompt attributes for backward compatibility."""
        # Map of prompt registry names to legacy attribute names
        legacy_mapping = {
            'first_pass': 'first_pass_prompt',
            'second_pass': 'second_pass_prompt',
            'global_check': 'global_check_prompt',
            'tagging': 'tagging_prompt',
            'relationship': 'relationship_prompt'
        }
        
        # For each mapping, try to get from registry and update attributes
        for registry_name, attr_name in legacy_mapping.items():
            try:
                model = self.model if self.use_function_calling else None
                template = self.prompt_registry.get(registry_name, model)
                
                # Set as direct attribute
                setattr(self, attr_name, template.template)
                
                # Set uppercase version in prompts namespace
                uppercase_name = attr_name.upper()
                setattr(self.prompts, uppercase_name, template.template)
                
            except KeyError:
                # Template not found in registry, ignore
                pass
    
    def get_prompt(self, name: str, **kwargs) -> str:
        """
        Get a formatted prompt by name.
        
        Args:
            name: Name of the prompt template
            **kwargs: Parameters to format the prompt with
            
        Returns:
            Formatted prompt string
        """
        model = self.model if self.use_function_calling else None
        
        try:
            # Try to get from the prompt registry first
            return self.prompt_registry.format(name, model=model, **kwargs)
        except KeyError:
            # Fallback to legacy prompt access
            attr_name = f"{name}_prompt"
            if hasattr(self, attr_name) and getattr(self, attr_name):
                template = getattr(self, attr_name)
                
                # Very basic template formatting
                for key, value in kwargs.items():
                    placeholder = "{" + key + "}"
                    template = template.replace(placeholder, str(value))
                
                return template
            
            # If still not found, raise error
            raise ValueError(f"Prompt template '{name}' not found")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "llm_backend": self.llm_backend,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "max_response_tokens": self.max_response_tokens,
            "window_size": self.window_size,
            "overlap_size": self.overlap_size,
            "tagging_enabled": self.tagging_enabled,
            "relationships_enabled": self.relationships_enabled,
            "track_transcript_positions": self.track_transcript_positions,
            "use_function_calling": self.use_function_calling,
            "function_schemas": self.function_schemas,
            "validate_function_output": self.validate_function_output,
            "ollama_url": self.ollama_url,
            "output_dir": self.output_dir,
            "debug": self.debug
        } 