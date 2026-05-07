from pypdf import PdfReader
import io

def pdf_text_extract(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text_chunks = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_chunks.append(text)

    return "\n".join(text_chunks)
