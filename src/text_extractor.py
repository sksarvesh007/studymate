from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path, page_number):
    reader = PdfReader(pdf_path)
    page = reader.pages[page_number]
    return page.extract_text()

