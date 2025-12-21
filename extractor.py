from PyPDF2 import PdfReader
import docx

def extract_text(file):
    filename = file.name.lower()

    # PDF
    if filename.endswith(".pdf"):
        reader = PdfReader(file)
        return " ".join(
            [page.extract_text() for page in reader.pages if page.extract_text()]
        )

    # DOCX
    elif filename.endswith(".docx"):
        doc = docx.Document(file)
        return " ".join([p.text for p in doc.paragraphs])

    # TXT or any text file
    else:
        return file.read().decode("utf-8")
