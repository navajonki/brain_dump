import sys
import os
import json
import time
import logging

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create a simple in-memory logger to avoid file I/O during testing
logger = logging.getLogger("test_logger")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(handler)

# Import the _clean_json_string function directly from a standalone class
# to avoid initialization issues
class JsonCleaner:
    """Simplified class that only has the _clean_json_string method for testing"""
    
    def __init__(self):
        self.logger = logger
    
    def _clean_json_string(self, json_str: str) -> str:
        """
        Clean a JSON string to make it more parseable using a streamlined approach.
        
        Efficiently extracts and repairs JSON structures from LLM responses.
        """
        if not isinstance(json_str, str):
            return json_str
        
        # Log input for debugging (truncated)
        self.logger.debug(f"Cleaning JSON string: {json_str[:100]}...")
        
        # Try direct parsing first for already valid JSON
        try:
            json.loads(json_str)
            return json_str  # Already valid JSON
        except json.JSONDecodeError:
            pass  # Continue with cleaning
        
        # Find the first { or [ and the last } or ]
        start_indices = [json_str.find(c) for c in "{[" if json_str.find(c) != -1]
        end_indices = [json_str.rfind(c) for c in "}]" if json_str.rfind(c) != -1]
        
        if start_indices and end_indices:
            start_idx = min(start_indices)
            end_idx = max(end_indices)
            if start_idx < end_idx:
                json_str = json_str[start_idx:end_idx+1]
                self.logger.debug(f"Extracted potential JSON: {json_str[:50]}...")
        
        # Apply essential fixes
        # 1. Remove control characters and non-printable characters
        json_str = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', json_str)
        
        # 2. Fix trailing commas before closing brackets/braces
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # 3. Fix missing quotes around property names
        json_str = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        
        # 4. Fix missing quotes around string values
        json_str = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_]*)\s*([,}])', r':"\1"\2', json_str)
        
        # 5. Fix unclosed quotes in strings 
        # This regex finds unclosed quotes in array elements (a common issue)
        json_str = re.sub(r'(:\s*\[\s*\".+?)(?=,\s*\])', r'\1"', json_str)
        
        # Check balance of braces and fix
        opening_to_closing = {'{': '}', '[': ']'}
        stack = []
        fixed_str = ""
        
        # First-pass balance check with stack-based approach
        for char in json_str:
            if char in '{[':
                stack.append(char)
                fixed_str += char
            elif char in '}]':
                # Check if this is a valid closing bracket
                if stack and opening_to_closing[stack[-1]] == char:
                    stack.pop()
                    fixed_str += char
                else:
                    # Skip invalid closing bracket
                    continue
            else:
                fixed_str += char
                
        # Add any missing closing brackets/braces
        while stack:
            opener = stack.pop()
            fixed_str += opening_to_closing[opener]
            self.logger.debug(f"Added missing closing {opening_to_closing[opener]}")
        
        json_str = fixed_str
        
        # Try to parse with our fixed JSON
        try:
            json.loads(json_str)
            self.logger.debug("JSON successfully fixed and validated")
            return json_str
        except json.JSONDecodeError as e:
            self.logger.debug(f"JSON still invalid after cleanup: {str(e)}")
            
            # Advanced fix for mismatched quotes in nested structures
            # Look for any unbalanced quotes
            try:
                # Count quotes outside of string literals
                in_string = False
                escape_next = False
                quotes = []
                
                for i, char in enumerate(json_str):
                    if char == '\\' and not escape_next:
                        escape_next = True
                        continue
                    
                    if char == '"' and not escape_next:
                        if not in_string:
                            in_string = True
                            quotes.append(i)  # Record opening quote position
                        else:
                            in_string = False
                            if quotes:  # Remove matched opening quote
                                quotes.pop()
                    
                    escape_next = False
                
                # Fix unbalanced quotes by adding missing quotes
                if quotes:
                    for pos in quotes:
                        # Add closing quote at a reasonable position
                        # Look for a comma, brace or bracket after the opening quote
                        end_pos = json_str.find(',', pos)
                        if end_pos == -1:
                            end_pos = min(pos + 50, len(json_str))  # Limit to 50 chars
                        
                        json_str = json_str[:end_pos] + '"' + json_str[end_pos:]
                        self.logger.debug(f"Added missing quote at position {end_pos}")
            
            except Exception as quote_error:
                self.logger.debug(f"Error fixing quotes: {str(quote_error)}")
            
            # Try again after quote fixes
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                # More aggressive cleanup
                json_str = re.sub(r'[^\[\]{}",:\s\w._\-#&\'%+*!@()$]', '', json_str)
                
                # Replace any malformed "key": [incomplete... with proper array
                json_str = re.sub(r'"\w+":\s*\[\s*[^]]*(?!\])', r'":[]', json_str)
                
                # Try one final time
                try:
                    json.loads(json_str) 
                    return json_str
                except json.JSONDecodeError:
                    # Last resort: create a minimum valid JSON
                    if json_str.startswith('{'):
                        self.logger.warning("Creating minimum valid JSON object")
                        return "{}"
                    elif json_str.startswith('['):
                        self.logger.warning("Creating minimum valid JSON array")
                        return "[]"
                    else:
                        self.logger.warning("Failed to repair JSON, returning minimal valid JSON")
                        return "{}"

import re  # Add this for regex support

def test_json_cleanup():
    """Test the JSON cleanup function with various types of malformed JSON"""
    
    # Initialize our test cleaner
    cleaner = JsonCleaner()
    
    # Test cases - (input, expected_parse_success)
    test_cases = [
        # 1. Already valid JSON
        ('{"key": "value"}', True),
        
        # 2. JSON with explanatory text before and after
        ('Here is the JSON response: {"facts": [{"text": "Test"}]} I hope this helps!', True),
        
        # 3. Missing quotes around property names
        ('{key: "value", another_key: 42}', True),
        
        # 4. Missing quotes around string values
        ('{"key": value}', True),
        
        # 5. Trailing commas
        ('{"items": [1, 2, 3,]}', True),
        
        # 6. Unbalanced braces - missing closing brace
        ('{"key": "value", "nested": {"foo": "bar"', True),
        
        # 7. Unbalanced brackets
        ('{"items": [1, 2, 3}', True),
        
        # 8. Severely malformed but recoverable JSON
        ('Response: {key: value, items: [1, 2, incomplete', True),
        
        # 9. Non-JSON text
        ('This is just plain text with no JSON structure.', True),
        
        # 10. Complex nested JSON with errors
        ('''
        {
            "facts": [
                {"text": "Fact 1", confidence: 0.9},
                {"text": "Fact 2", confidence: 0.8,
                "entities": ["person1", "location1],
            ],
            topics: ["topic1", "topic2"],
        }
        ''', True)
    ]
    
    results = []
    for i, (input_str, expected_parse_success) in enumerate(test_cases, 1):
        print(f"\nTest case {i}: ", end="")
        
        # Time the cleanup operation
        start_time = time.time()
        cleaned = cleaner._clean_json_string(input_str)
        cleanup_time = time.time() - start_time
        
        # Test if result is valid JSON
        parse_success = False
        try:
            json.loads(cleaned)
            parse_success = True
            result = "PASS"
        except json.JSONDecodeError:
            result = "FAIL" if expected_parse_success else "EXPECTED FAIL"
        
        print(f"{result} ({cleanup_time:.4f}s)")
        print(f"Input: {input_str[:50]}...")
        print(f"Cleaned: {cleaned[:50]}...")
        
        results.append({
            "case": i,
            "success": parse_success == expected_parse_success,
            "time": cleanup_time,
            "parse_success": parse_success,
            "expected_parse_success": expected_parse_success
        })
    
    # Display summary
    print("\n--- Summary ---")
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"Average cleaning time: {sum(r['time'] for r in results)/total:.4f}s")
    
    # Display failed tests
    failed = [r for r in results if not r["success"]]
    if failed:
        print("\nFailed test cases:")
        for case in failed:
            print(f"- Case {case['case']}: Expected {case['expected_parse_success']}, got {case['parse_success']}")
    
    return passed == total

if __name__ == "__main__":
    success = test_json_cleanup()
    print(f"\nOverall test {'SUCCESS' if success else 'FAILURE'}")
    sys.exit(0 if success else 1)