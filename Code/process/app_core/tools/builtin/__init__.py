"""Built-in tools loaded by the provider-neutral registry."""

from .base import BaseTool
from .todo_list import Tool as TodoListTool
from .scientific_calculator import Tool as ScientificCalculatorTool
try:
    from .pdf_processor import Tool as PdfExtractorTool
except ImportError:
    PdfExtractorTool = None


def iter_tools():
    tools = [TodoListTool({}, {}), ScientificCalculatorTool({}, {})]
    if PdfExtractorTool is not None:
        tools.append(PdfExtractorTool({}, {}))
    return tools


__all__ = ["BaseTool", "iter_tools"]
