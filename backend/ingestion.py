import fitz
import pytesseract
from PIL import Image
import io

# Set Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)

    full_text = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        # Try normal text extraction
        text = page.get_text()

        # If scanned page
        if not text.strip():
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")

            image = Image.open(io.BytesIO(img_bytes))

            text = pytesseract.image_to_string(image)

        full_text.append({
            "page": page_num + 1,
            "text": text
        })

    return full_text