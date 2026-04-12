import re
from typing import List, Dict


def chunk_text(
    text: str,
    filename: str,
    document_id: str = "",
    chunk_size: int = 1050,
    overlap: int = 240,
) -> List[Dict]:
    """
    Split PDF text into semantic chunks while preserving section structure.

    Design goals:
    - Keep headings attached to their related body text.
    - Respect paragraph boundaries whenever possible.
    - Use larger chunk windows and overlap for better retrieval context.
    """
    safe_chunk_size = max(900, min(1200, int(chunk_size)))
    safe_overlap = max(200, min(300, int(overlap)))

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [b.strip() for b in re.split(r"\n{2,}", normalized) if b.strip()]
    if not blocks:
        return []

    sections = _build_sections(blocks)
    segmented_sections = _split_oversized_sections(sections, safe_chunk_size)
    chunk_texts = _pack_sections(segmented_sections, safe_chunk_size)
    return _with_overlap(chunk_texts, safe_overlap, filename, document_id)


def _build_sections(blocks: List[str]) -> List[Dict]:
    sections: List[Dict] = []
    pending_heading = ""

    for block in blocks:
        if _is_heading(block):
            pending_heading = block
            continue

        if pending_heading:
            combined = f"{pending_heading}\n{block}"
            sections.append({"text": combined, "heading": pending_heading})
            pending_heading = ""
        else:
            sections.append({"text": block, "heading": ""})

    if pending_heading:
        sections.append({"text": pending_heading, "heading": pending_heading})

    return sections


def _split_oversized_sections(sections: List[Dict], chunk_size: int) -> List[Dict]:
    output: List[Dict] = []
    sentence_end = re.compile(r"(?<=[.!?])\s+")

    for section in sections:
        text = section["text"]
        heading = section["heading"]

        if len(text) <= chunk_size:
            output.append(section)
            continue

        content = text
        if heading and text.startswith(heading):
            content = text[len(heading):].strip()

        sentences = [s.strip() for s in sentence_end.split(content) if s.strip()]
        if not sentences:
            output.append(section)
            continue

        buffer = heading if heading else ""
        for sentence in sentences:
            candidate = f"{buffer} {sentence}".strip() if buffer else sentence
            if len(candidate) <= chunk_size:
                buffer = candidate
            else:
                if buffer:
                    output.append({"text": buffer.strip(), "heading": heading})
                buffer = f"{heading}\n{sentence}".strip() if heading else sentence
        if buffer:
            output.append({"text": buffer.strip(), "heading": heading})

    return output


def _pack_sections(sections: List[Dict], chunk_size: int) -> List[Dict]:
    packed: List[Dict] = []
    buffer_text = ""
    buffer_heading = ""

    for section in sections:
        section_text = section["text"].strip()
        section_heading = section["heading"]

        if not buffer_text:
            buffer_text = section_text
            buffer_heading = section_heading
            continue

        candidate = f"{buffer_text}\n\n{section_text}"
        if len(candidate) <= chunk_size:
            buffer_text = candidate
        else:
            packed.append({"text": buffer_text.strip(), "heading": buffer_heading})
            buffer_text = section_text
            buffer_heading = section_heading

    if buffer_text:
        packed.append({"text": buffer_text.strip(), "heading": buffer_heading})

    return packed


def _with_overlap(chunk_texts: List[Dict], overlap: int, filename: str, document_id: str) -> List[Dict]:
    chunks: List[Dict] = []
    for i, chunk_data in enumerate(chunk_texts):
        body = chunk_data["text"].strip()
        heading = chunk_data.get("heading", "")
        if i > 0 and overlap > 0:
            overlap_text = _tail_overlap(chunk_texts[i - 1]["text"], overlap)
            if overlap_text and not body.startswith(overlap_text):
                body = f"{overlap_text}\n{body}".strip()
        chunks.append(
            {
                "text": body,
                "filename": filename,
                "document_id": document_id,
                "chunk_index": i,
                "heading": heading,
            }
        )
    return chunks


def _tail_overlap(text: str, overlap_chars: int) -> str:
    tail = text[-overlap_chars:].strip()
    if not tail:
        return ""
    # Prefer starting overlap from a sentence boundary to reduce broken context.
    start = max(tail.rfind(". "), tail.rfind("? "), tail.rfind("! "))
    if start != -1 and start + 2 < len(tail):
        return tail[start + 2 :].strip()
    return tail


def _is_heading(block: str) -> bool:
    single_line = " ".join(block.splitlines()).strip()
    if not single_line:
        return False
    if len(single_line) > 120:
        return False
    if single_line.endswith((".", ";", ",")):
        return False

    has_number_prefix = bool(re.match(r"^\d+(\.\d+)*[\)\.]?\s+\S+", single_line))
    mostly_title_case = sum(1 for w in single_line.split() if w[:1].isupper()) >= max(1, len(single_line.split()) // 2)
    short_line = len(single_line.split()) <= 12
    return has_number_prefix or (short_line and mostly_title_case)
