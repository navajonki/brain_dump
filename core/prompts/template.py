"""
PromptTemplate class for the centralized prompt system.
"""
from typing import List, Dict, Any, Optional, Set
import re
import string


class PromptTemplate:
    """
    A class representing a prompt template with metadata.
    
    Attributes:
        name: Unique identifier for the template
        template: The template string with placeholders
        description: Human-readable description of the template's purpose
        required_params: List of parameters that must be provided when formatting
        model_specific: Whether this template is specific to a particular model
        model: The model this template is designed for (if model_specific is True)
        version: Version of the template for tracking changes
        tags: List of tags for categorizing templates
    """
    
    def __init__(
        self,
        name: str,
        template: str,
        description: Optional[str] = None,
        required_params: Optional[List[str]] = None,
        model: Optional[str] = None,
        version: str = "1.0",
        tags: Optional[List[str]] = None
    ):
        """Initialize a prompt template."""
        self.name = name
        self.template = template
        self.description = description or f"Template for {name}"
        self.required_params = required_params or []
        self.model_specific = model is not None
        self.model = model
        self.version = version
        self.tags = tags or []
        
        # Extract parameters from the template
        self._extract_parameters()
    
    def _extract_parameters(self) -> None:
        """
        Extract parameters from the template string and update required_params.
        
        This uses a regex to find all format string placeholders like {param_name}.
        """
        # Find all parameters in the format {param_name}
        params = re.findall(r'\{([a-zA-Z0-9_]+)\}', self.template)
        
        # Add to required_params if not already there
        for param in params:
            if param not in self.required_params:
                self.required_params.append(param)
    
    def format(self, **kwargs) -> str:
        """
        Format the template with the provided parameters.
        
        Args:
            **kwargs: Parameters to fill in the template
            
        Returns:
            The formatted prompt string
            
        Raises:
            ValueError: If a required parameter is missing
        """
        # Check for missing required parameters
        missing_params = [param for param in self.required_params if param not in kwargs]
        if missing_params:
            raise ValueError(f"Missing required parameters for {self.name} prompt: {', '.join(missing_params)}")
        
        # Format the template
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Unknown parameter in template {self.name}: {str(e)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the template to a dictionary for serialization."""
        return {
            "name": self.name,
            "template": self.template,
            "description": self.description,
            "required_params": self.required_params,
            "model": self.model,
            "model_specific": self.model_specific,
            "version": self.version,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptTemplate':
        """Create a template from a dictionary."""
        return cls(
            name=data["name"],
            template=data["template"],
            description=data.get("description"),
            required_params=data.get("required_params"),
            model=data.get("model"),
            version=data.get("version", "1.0"),
            tags=data.get("tags")
        )
    
    def __str__(self) -> str:
        """String representation of the template."""
        return f"PromptTemplate({self.name}, version={self.version})"
    
    def __repr__(self) -> str:
        """Detailed representation of the template."""
        return f"PromptTemplate(name='{self.name}', model='{self.model}', params={self.required_params})"