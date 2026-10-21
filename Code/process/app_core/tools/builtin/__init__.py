"""Built-in tools loaded by the provider-neutral registry. Each call runs in a disposable worker (tools/worker.py) that
imports only that tool's module, so this package imports none of them (or pypdf) up front."""

from .base import BaseTool


def iter_tools(todo_directory=None):
    """todo_directory: where todo_list keeps its list (DataPaths.todo_list); its worker receives it in the tool's config."""
    from .todo_list import Tool as TodoListTool
    from .scientific_calculator import Tool as ScientificCalculatorTool
    tools = [TodoListTool({'data_directory': str(todo_directory)} if todo_directory else {}, {}), ScientificCalculatorTool({}, {})]
    try:
        from .pdf_processor import Tool as PdfExtractorTool
    except ImportError:
        return tools
    tools.append(PdfExtractorTool({}, {}))
    return tools


__all__ = ["BaseTool", "iter_tools"]
