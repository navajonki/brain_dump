from typing import Dict, Any

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
        
        # Prompts configuration - support both uppercase and lowercase prompt names
        self.prompts = type('Prompts', (), {})()
        
        # Load prompts from kwargs
        # First check for lowercase prompt names (from YAML)
        self.first_pass_prompt = kwargs.get('first_pass_prompt', '')
        self.second_pass_prompt = kwargs.get('second_pass_prompt', '')
        self.global_check_prompt = kwargs.get('global_check_prompt', '')
        self.tagging_prompt = kwargs.get('tagging_prompt', '')
        self.relationship_prompt = kwargs.get('relationship_prompt', '')
        
        # Also set them in the prompts namespace with uppercase names for backward compatibility
        self.prompts.FIRST_PASS_PROMPT = self.first_pass_prompt
        self.prompts.SECOND_PASS_PROMPT = self.second_pass_prompt
        self.prompts.GLOBAL_CHECK_PROMPT = self.global_check_prompt
        self.prompts.TAGGING_PROMPT = self.tagging_prompt
        self.prompts.RELATIONSHIP_PROMPT = self.relationship_prompt
        
        # Function schemas
        self.function_schemas = kwargs.get('function_schemas', {})
        
        # Ollama settings
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Output settings
        self.output_dir = "output/chunks"
        self.debug = True
        
        # Load custom prompts if a different module is specified
        if 'prompt_template_module' in kwargs:
            self.prompt_template_module = kwargs.get('prompt_template_module')
            self._load_prompts()
    
    def _load_prompts(self):
        """Load prompt templates from the configured module."""
        try:
            if not hasattr(self, 'prompt_template_module') or not self.prompt_template_module:
                return
                
            module_path = self.prompt_template_module.replace(".", "/") + ".py"
            with open(module_path, "r") as f:
                module_code = f.read()
                
            # Create a new module
            import types
            module = types.ModuleType(self.prompt_template_module)
            
            # Execute the module code
            exec(module_code, module.__dict__)
            
            # Store prompts
            if hasattr(module, 'FIRST_PASS_PROMPT'):
                self.first_pass_prompt = module.FIRST_PASS_PROMPT
                self.prompts.FIRST_PASS_PROMPT = module.FIRST_PASS_PROMPT
            
            if hasattr(module, 'SECOND_PASS_PROMPT'):
                self.second_pass_prompt = module.SECOND_PASS_PROMPT
                self.prompts.SECOND_PASS_PROMPT = module.SECOND_PASS_PROMPT
            
            if hasattr(module, 'TAGGING_PROMPT'):
                self.tagging_prompt = module.TAGGING_PROMPT
                self.prompts.TAGGING_PROMPT = module.TAGGING_PROMPT
            
            if hasattr(module, 'RELATIONSHIP_PROMPT'):
                self.relationship_prompt = module.RELATIONSHIP_PROMPT
                self.prompts.RELATIONSHIP_PROMPT = module.RELATIONSHIP_PROMPT
            
            if hasattr(module, 'GLOBAL_CHECK_PROMPT'):
                self.global_check_prompt = module.GLOBAL_CHECK_PROMPT
                self.prompts.GLOBAL_CHECK_PROMPT = module.GLOBAL_CHECK_PROMPT
            
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to load prompt templates from {self.prompt_template_module}: {str(e)}")
    
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