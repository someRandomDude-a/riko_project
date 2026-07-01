import math
from typing import Any
from .base import BaseTool, ToolType


class Tool(BaseTool):
    TOOL_NAME = "scientific_calculator"
    TOOL_DESCRIPTION = "Evaluates mathematical expressions using Python's math functions."
    TOOL_TYPE = ToolType.FUNCTION

    MCP_PROMPT = """scientific_calculator:
  Evaluates a mathematical expression using Python's math functions.
  
  Parameters:
    expression (str) - The mathematical expression to evaluate. 
                       Use standard arithmetic operators and math functions.
  
  Supported functions and constants:
    sin, cos, tan, asin, acos, atan, atan2, 
    log (natural), log10, log2,
    exp, sqrt, pow, 
    pi, e, tau, inf, nan
  
  Basic arithmetic: +, -, *, /, //, ** (exponentiation), % (modulo)
  Parentheses for grouping: ( ... )
  Use ** for exponentiation (e.g., 2**10 = 1024)
  
  Examples:
    scientific_calculator(expression="sin(pi/4) + log(100)")  -> 1.7071067811865475
    scientific_calculator(expression="sqrt(2**10)")           -> 32.0
    scientific_calculator(expression="5 * (3 + 2) - 4")       -> 21.0
    scientific_calculator(expression="atan2(3, 4) * 180 / pi") -> 36.86989764584402
    scientific_calculator(expression="pow(2, 8)")             -> 256.0
    scientific_calculator(expression="log10(1000)")           -> 3.0
    scientific_calculator(expression="exp(1)")                -> 2.718281828459045
  
  If the expression is invalid (syntax error, undefined function, etc.), 
  the tool returns an error message describing the problem.
"""

    def _call(self, expression: str) -> str: #type: ignore
        """
        Evaluate the mathematical expression safely using the math module.
        Returns the result as a string for easy display.
        """
        # Build a safe evaluation environment: only math functions and constants
        safe_dict = {
            # Mathematical functions
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'atan2': math.atan2,
            'log': math.log,
            'log10': math.log10,
            'log2': math.log2,
            'exp': math.exp,
            'sqrt': math.sqrt,
            'pow': math.pow,
            'hypot': math.hypot,
            # Constants
            'pi': math.pi,
            'e': math.e,
            'tau': math.tau,
            'inf': math.inf,
            'nan': math.nan,
        }
        # Disable builtins for safety (prevent code injection)
        safe_dict['__builtins__'] = None

        try:
            # Evaluate the expression with the restricted namespace
            result = eval(expression, safe_dict, {})
            # Format the result: convert to string with repr for precision
            return repr(result)
        except Exception as e:
            # Return a user-friendly error message
            return f"Error evaluating expression: {e}"