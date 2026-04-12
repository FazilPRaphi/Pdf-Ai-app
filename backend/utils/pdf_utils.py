import fitz  # PyMuPDF


def extract_text_from_pdf(file_path: str):
    """
    Extract text from PDF and return:
    - full text
    - total pages
    """
    try:
        text_parts = []
        page_count = 0
        empty_pages = 0
        failed_pages = 0

        with fitz.open(file_path) as doc:
            page_count = len(doc)

            for page_index in range(page_count):
                try:
                    page_text = doc.load_page(page_index).get_text("text")
                except Exception:
                    failed_pages += 1
                    continue

                cleaned = page_text.strip()
                if cleaned:
                    text_parts.append(cleaned)
                else:
                    empty_pages += 1

        full_text = "\n\n".join(text_parts).strip()

        return {
            "success": True,
            "text": full_text,
            "pages": page_count,
            "empty_pages": empty_pages,
            "failed_pages": failed_pages,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
