import sys
import os
from pathlib import Path
import unittest
import json

# Add project root to Python path
project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

from core.prompts import template_registry, PromptTemplate
from core.prompts.defaults import DEFAULT_PROMPTS, MODEL_SPECIFIC_PROMPTS
from config.chunking.chunking_config import ChunkingConfig

# Define a mock LLM response for integration testing
class MockLLMResponse:
    def json(self):
        return {"choices": [{"message": {"content": "This is a mock response"}}]}
    
    @property
    def text(self):
        return "This is a mock response"

# Mock LLM backend for testing
class MockLLMBackend:
    def __init__(self, *args, **kwargs):
        self.prompt_history = []
    
    def generate(self, prompt, **kwargs):
        self.prompt_history.append(prompt)
        return "This is a mock response"
    
    def generate_json(self, prompt, **kwargs):
        self.prompt_history.append(prompt)
        return {"message": "This is a mock response"}


class TestPromptSystem(unittest.TestCase):
    """Test the centralized prompt template system."""
    
    def setUp(self):
        """Set up test environment."""
        # Reset the template registry before each test
        template_registry._templates = {"default": {}}
        template_registry._initialized = False
        template_registry.load_defaults()
    
    def test_registry_initialization(self):
        """Test that the registry initializes correctly."""
        # Check that default templates are loaded
        self.assertTrue(template_registry.has_template("first_pass"))
        self.assertTrue(template_registry.has_template("second_pass"))
        self.assertTrue(template_registry.has_template("global_check"))
        self.assertTrue(template_registry.has_template("tagging"))
        self.assertTrue(template_registry.has_template("relationship"))
        
        # Check that model-specific templates are loaded
        self.assertTrue(template_registry.has_template("first_pass", model="mistral"))
    
    def test_template_formatting(self):
        """Test template formatting with parameters."""
        # Get a template and format it
        formatted = template_registry.format(
            "first_pass", 
            window_text="This is a test transcript."
        )
        
        # Check that the parameter was properly filled in
        self.assertIn("This is a test transcript.", formatted)
        
        # Test with model-specific template
        formatted_mistral = template_registry.format(
            "first_pass", 
            model="mistral", 
            window_text="This is a model-specific test."
        )
        
        # Check that the model-specific template was used
        self.assertIn("This is a model-specific test.", formatted_mistral)
        self.assertIn("[AVAILABLE_TOOLS]", formatted_mistral)
    
    def test_template_parameter_validation(self):
        """Test that templates validate their parameters."""
        # Test with missing parameter
        with self.assertRaises(ValueError):
            template_registry.format("first_pass")
        
        # Test with wrong parameter
        with self.assertRaises(ValueError):
            template_registry.format("first_pass", wrong_param="test")
    
    def test_chunking_config_integration(self):
        """Test integration with ChunkingConfig."""
        # Create a config
        config = ChunkingConfig(
            model="gpt-3.5-turbo",
            use_function_calling=False
        )
        
        # Get a prompt using the config
        prompt = config.get_prompt("first_pass", window_text="Testing ChunkingConfig integration.")
        
        # Check that the prompt was formatted correctly
        self.assertIn("Testing ChunkingConfig integration.", prompt)
        
        # Test with a custom prompt override
        custom_prompt = "Custom prompt with {window_text}"
        config = ChunkingConfig(
            model="gpt-3.5-turbo",
            first_pass_prompt=custom_prompt
        )
        
        # Get the overridden prompt
        prompt = config.get_prompt("first_pass", window_text="Custom override test.")
        
        # Check that the override was used
        self.assertEqual(prompt, "Custom prompt with Custom override test.")
    
    def test_template_registration(self):
        """Test registering new templates."""
        # Register a new template
        template_registry.register(
            name="test_template",
            template="Test template with {param1} and {param2}",
            description="Test template",
            required_params=["param1", "param2"]
        )
        
        # Check that it was registered
        self.assertTrue(template_registry.has_template("test_template"))
        
        # Format the template
        formatted = template_registry.format("test_template", param1="value1", param2="value2")
        self.assertEqual(formatted, "Test template with value1 and value2")
        
        # Try to register a duplicate template
        with self.assertRaises(ValueError):
            template_registry.register(
                name="test_template",
                template="Duplicate template"
            )
        
        # Register with override
        template_registry.register(
            name="test_template",
            template="Overridden template with {param}",
            required_params=["param"],
            override=True
        )
        
        # Check that it was overridden
        formatted = template_registry.format("test_template", param="override_value")
        self.assertEqual(formatted, "Overridden template with override_value")


class TestChunkerIntegration(unittest.TestCase):
    """Test the integration of the prompt system with the AtomicChunker."""
    
    def setUp(self):
        """Set up test environment."""
        # Reset the template registry before each test
        template_registry._templates = {"default": {}}
        template_registry._initialized = False
        template_registry.load_defaults()
        
        # Create a test transcript
        self.test_transcript = """
        This is a test transcript. It contains multiple sentences.
        The speaker is talking about various topics.
        This is a simple test to verify prompt template usage.
        """
        
        # Patch the LLM backend creation function to use our mock
        import core.llm_backends
        self.original_create_llm_backend = core.llm_backends.create_llm_backend
        core.llm_backends.create_llm_backend = lambda *args, **kwargs: MockLLMBackend()
    
    def tearDown(self):
        """Clean up after tests."""
        # Restore the original LLM backend creation function
        import core.llm_backends
        core.llm_backends.create_llm_backend = self.original_create_llm_backend
    
    def test_chunker_prompt_usage(self):
        """Test that the AtomicChunker uses the centralized prompt system."""
        # Import here to avoid circular imports with the patching
        from core.chunking import TextChunker
        
        # Create a config with a custom first pass prompt
        custom_prompt = "Custom prompt with {window_text}"
        config = ChunkingConfig(
            model="test-model",
            first_pass_prompt=custom_prompt,
            llm_backend="test"  # This will use our mock
        )
        
        # Create a chunker with the config
        chunker = TextChunker(config)
        
        # Check that the prompt registry was properly set up
        self.assertTrue(chunker.config.prompt_registry.has_template("first_pass"))
        
        # Spy on the _call_llm method to capture prompt usage
        original_call_llm = chunker._call_llm
        call_history = []
        
        def mock_call_llm(prompt, *args, **kwargs):
            call_history.append(prompt)
            return {"facts": [{"text": "Test fact", "confidence": 0.9}]}
        
        chunker._call_llm = mock_call_llm
        
        # Call the first pass method directly to test prompt usage
        chunker._extract_atomic_facts_first_pass("Test window text", 1)
        
        # Check that the custom prompt was used
        self.assertEqual(len(call_history), 1)
        self.assertEqual(call_history[0], "Custom prompt with Test window text")
        
        # Restore original method
        chunker._call_llm = original_call_llm


if __name__ == "__main__":
    unittest.main()