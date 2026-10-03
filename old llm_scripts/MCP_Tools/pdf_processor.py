import io
from pypdf import PdfReader
from .base import BaseTool, ToolType


class Tool(BaseTool):
    TOOL_NAME = "pdf_extractor"
    TOOL_DESCRIPTION = "Extracts all text content from a PDF file sent by the user"
    TOOL_TYPE = ToolType.RESOURCE

    MCP_PROMPT = """pdf_extractor (resource):
  This tool automatically processes PDF files attached by the user.
"""

    def _call(self, file_bytes: bytes) -> str: # type: ignore
        """
        Extract all text from a PDF given as bytes.
        Called automatically by the system when a PDF is attached.
        """
        if not isinstance(file_bytes, bytes):
            raise TypeError("file_bytes must be a bytes object")

        try:
            reader = PdfReader(io.BytesIO(file_bytes))
        except Exception as e:
            raise ValueError(f"Invalid or corrupted PDF file: {e}")

        if len(reader.pages) == 0:
            return ""

        text_chunks = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_chunks.append(text)

        result = "\n".join(text_chunks)
        result = result[:100000]  # limit max size
        return f"\n[PDF CONTENT START]\n{result}\n[PDF CONTENT END]\n"