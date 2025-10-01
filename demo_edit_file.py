# Demo file for editing demonstrations
def hello_world():
    """A simple hello world function with enhanced greeting."""
    print("Hello, World! Welcome to the tool demonstrations!")

def calculate_sum(a, b):
    """Calculate the sum of two numbers with validation."""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise ValueError("Both arguments must be numbers")
    return a + b

class DemoClass:
    """A demo class for editing examples."""
    
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        return f"Hello, {self.name}! Nice to meet you!"