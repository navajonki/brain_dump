# Try importing each package
try:
    import tiktoken
    print("tiktoken imported successfully")
except ImportError as e:
    print(f"Failed to import tiktoken: {e}")

try:
    import openai
    print("openai imported successfully")
except ImportError as e:
    print(f"Failed to import openai: {e}") 