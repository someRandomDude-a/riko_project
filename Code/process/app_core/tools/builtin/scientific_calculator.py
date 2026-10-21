import ast
import math
import operator
from .base import BaseTool

# Expressions come from model output, which prompt injection can control, so they are never
# passed to eval(). Only the syntax below is evaluated; everything else is rejected.
FUNCTIONS = {name: getattr(math, name) for name in (
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2', 'log', 'log10', 'log2', 'exp', 'sqrt', 'pow', 'hypot')}
CONSTANTS = {'pi': math.pi, 'e': math.e, 'tau': math.tau, 'inf': math.inf, 'nan': math.nan}
BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
          ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
MAX_LENGTH = 1000  # characters
MAX_DEPTH = 100  # nested operations
MAX_INT_BITS = 10_000  # ~3,000 digits; checked before exponentiation, so 9**9**9 cannot hang


def _evaluate(node, depth=0):
    if depth > MAX_DEPTH: raise ValueError('Expression is nested too deeply')
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name) and node.id in CONSTANTS:
        return CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
        return _checked(UNARY[type(node.op)](_evaluate(node.operand, depth + 1)))
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
        if isinstance(node.op, ast.Pow):  # Right-associative: recurse.
            return _binary(node.op, _evaluate(node.left, depth + 1), _evaluate(node.right, depth + 1))
        # a + b + c ... nests leftwards; walk such chains iteratively so a long flat sum or
        # product is not mistaken for deep nesting.
        chain = []
        while isinstance(node, ast.BinOp) and type(node.op) in BINARY and not isinstance(node.op, ast.Pow):
            chain.append(node)
            node = node.left
        value = _evaluate(node, depth + 1)
        for link in reversed(chain): value = _binary(link.op, value, _evaluate(link.right, depth + 1))
        return value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS and not node.keywords:
        arguments = [_evaluate(argument, depth + 1) for argument in node.args]
        try: value = FUNCTIONS[node.func.id](*arguments)
        except ValueError as exc: raise ValueError(f'Math domain error in {node.func.id}(): {exc}') from None
        return _checked(value)
    raise ValueError(_describe(node))


def _binary(op, left, right):
    # Refuse only powers that are certainly too large (|left| >= 2**(bits-1)); anything that
    # passes has under 2 * MAX_INT_BITS bits, computes in microseconds, and _checked enforces
    # the exact limit, so 2**9998 and 2**4999 * 2**4999 agree.
    if isinstance(op, ast.Pow) and isinstance(left, int) and isinstance(right, int) \
            and abs(left) > 1 and right * (left.bit_length() - 1) >= MAX_INT_BITS:
        raise ValueError('Result is too large')
    return _checked(BINARY[type(op)](left, right))


def _describe(node):
    if isinstance(node, ast.Name):
        return f"Unknown name '{node.id}'; available constants: {', '.join(CONSTANTS)}"
    if isinstance(node, ast.Call):
        name = node.func.id if isinstance(node.func, ast.Name) else None
        if name in FUNCTIONS: return f'{name}() takes positional arguments only'
        return f"Unknown function{f' {name!r}' if name else ''}; available functions: {', '.join(FUNCTIONS)}"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitXor): return "'^' is not supported; use ** for powers"
    if isinstance(node, ast.Constant): return f'Only numbers are supported, not {type(node.value).__name__}'
    return f'Unsupported syntax ({type(node).__name__}); use numbers, + - * / // % **, parentheses, constants and math functions'


def _checked(value):
    if isinstance(value, complex): raise ValueError('Result is a complex number')
    if isinstance(value, int) and value.bit_length() > MAX_INT_BITS: raise ValueError('Result is too large')
    return value


def evaluate(expression):
    if not isinstance(expression, str) or not expression.strip(): raise ValueError('Provide an expression')
    if len(expression) > MAX_LENGTH: raise ValueError(f'Expression is longer than {MAX_LENGTH} characters')
    # The parser reports overly complex input as MemoryError or RecursionError, sometimes with no message.
    try: tree = ast.parse(expression.strip(), mode='eval')
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        raise ValueError(f'Invalid expression: {str(exc) or "too complex to parse"}') from None
    try: return _evaluate(tree.body)
    except (ArithmeticError, TypeError, RecursionError) as exc: raise ValueError(f'Error evaluating expression: {exc}') from None


class Tool(BaseTool):
    TOOL_NAME = "scientific_calculator"
    TOOL_DESCRIPTION = "Evaluates mathematical expressions using Python's math functions."

    def _call(self, expression: str) -> str: #type: ignore
        """Evaluate the expression with the restricted evaluator; errors raise ValueError."""
        return repr(evaluate(expression))
