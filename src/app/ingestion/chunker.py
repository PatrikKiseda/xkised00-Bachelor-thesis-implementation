"""
Author: Patrik Kiseda
File: src/app/ingestion/chunker.py
Description: Recursive character chunking with overlap for deterministic indexing.
"""

from __future__ import annotations


SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", " ", "")

def chunk_text_recursive(
    text: str,
    *,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
) -> list[str]:
    """Split text recursively using separators, then apply overlap.

    Args:
        text: Text to split.
        chunk_size_chars: Final max-ish chunk size in chars.
        chunk_overlap_chars: Chars copied from previous chunk.

    Returns:
        List of chunks ready for indexing.
    """
    # Public entrypoint used by indexing: split text, apply overlap.
    if chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be greater than 0.")
    if chunk_overlap_chars < 0:
        raise ValueError("chunk_overlap_chars must be greater than or equal to 0.")
    if chunk_overlap_chars >= chunk_size_chars:
        raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars.")

    normalized = text.strip()
    if not normalized:
        return []
    if len(normalized) <= chunk_size_chars:
        return [normalized]

    # First create a "base" that does not overlap with chunks that are slightly smaller.
    # Then overlaps are added in a second pass, this way we can keep a simpler logic for the recursive splitting.
    base_chunk_size = chunk_size_chars - chunk_overlap_chars
    base_chunks = _split_recursive(
        normalized,
        separators=SEPARATORS,
        chunk_size_chars=base_chunk_size,
    )    
    cleaned_base_chunks = [chunk.strip() for chunk in base_chunks if chunk.strip()]
    if chunk_overlap_chars == 0 or len(cleaned_base_chunks) <= 1:
        return cleaned_base_chunks
    return _apply_overlap(cleaned_base_chunks, chunk_overlap_chars)

# The rest of the functions are internal helpers for the recursive splitting logic.
def _split_recursive(
    text: str,
    *,
    separators: tuple[str, ...],
    chunk_size_chars: int,
) -> list[str]:
    """Recursively split text until chunks fit into the target size.

    Args:
        text: Text to split.
        separators: Separators to try, from coarse to fine.
        chunk_size_chars: Target chunk size in characters.

    Returns:
        Base chunks without overlap.
    """
    # Recursive splitter: keep trying finer separators until pieces fit.
    separator = _pick_separator(text, separators)
    if separator == "":
        return _hard_split(text, chunk_size_chars)

    separator_index = separators.index(separator)
    next_separators = separators[separator_index + 1 :]

    splits = _split_keep_separator(text, separator)
    current_group: list[str] = []
    chunks: list[str] = []

    # Process each split piece: small enough, keep it; too big, either recurse call or hard-split.
    for split in splits:
        if len(split) <= chunk_size_chars:
            # Keep accumulating nearby splits while they still fit together.
            current_group.append(split)
            continue

        if current_group:
            # Get a chunk out of the group of accumulated splits before addressing a big split.
            chunks.extend(
                _merge_splits(
                    current_group,
                    chunk_size_chars=chunk_size_chars,
                )
            )
            current_group = []

        if next_separators:
            # This piece is still too big -> recurse with a more fine-grained separator.
            # This is the main idea of the recursive logic that allows us to find natural chunk boundaries in this algorithm.
            chunks.extend(
                _split_recursive(
                    split,
                    separators=next_separators,
                    chunk_size_chars=chunk_size_chars,
                )
            )
        else:
            # No separators left, so we hard-cut by size.
            chunks.extend(_hard_split(split, chunk_size_chars))

    if current_group:
        # Flush trailing pieces at the end of the loop.
        chunks.extend(
            _merge_splits(
                current_group,
                chunk_size_chars=chunk_size_chars,
            )
        )
    return chunks


def _pick_separator(text: str, separators: tuple[str, ...]) -> str:
    """Choose the first separator that actually appears in the text.

    Args:
        text: Text to inspect.
        separators: Candidate separators.

    Returns:
        Matching separator, or empty string for hard split.
    """
    # Choose the first separator that actually appears in the text.
    # Empty string is the fallback that means "just hard split".
    for separator in separators:
        if separator == "":
            return separator
        if separator in text:
            return separator
    return ""


def _split_keep_separator(text: str, separator: str) -> list[str]:
    """Split text while keeping separators attached to previous pieces.

    Args:
        text: Text to split.
        separator: Separator to split on.

    Returns:
        Split pieces with separators preserved.
    """
    # Split text but keep separators attached to previous pieces.
    # That preserves punctuation/newline context better than dropping them.
    if not separator:
        return list(text)

    splits: list[str] = []
    start = 0
    separator_len = len(separator)

    # Loop through text and split by separator, while attached to the previous piece.
    while True:
        index = text.find(separator, start)
        if index == -1:
            tail = text[start:]
            if tail:
                splits.append(tail)
            break

        end = index + separator_len
        piece = text[start:end]
        if piece:
            splits.append(piece)
        start = end

    return splits

def _merge_splits(
    splits: list[str],
    *,
    chunk_size_chars: int,
) -> list[str]:
    """Greedily merge nearby pieces until they reach the chunk size.

    Args:
        splits: Split pieces to merge.
        chunk_size_chars: Target chunk size.

    Returns:
        Merged chunks.
    """
    merged: list[str] = []
    current = ""

    # Merge splits, if a split is too big and its alone, it will be recursivelly split, 
    # or hard-split if no separators are left, so we can just append it to the merged output.
    for split in splits:
        if not split:
            continue
        if len(split) > chunk_size_chars:
            if current and current.strip():
                merged.append(current)
            current = ""
            merged.extend(_hard_split(split, chunk_size_chars))
            continue

        # If the current piece plus the next split fits within the chunk size, merge them; otherwise, 
        if not current:
            current = split
            continue

        candidate = current + split
        if len(candidate) <= chunk_size_chars:
            current = candidate
            continue

        # Flush current to merged and start a new current with the next split.
        if current.strip():
            merged.append(current)
        current = split

    # Flush any trailing current piece after the loop.
    if current and current.strip():
        merged.append(current)

    return merged

def _hard_split(text: str, chunk_size_chars: int) -> list[str]:
    """Last-resort fixed-size splitter when separators are not useful.

    Args:
        text: Text to split.
        chunk_size_chars: Max chunk size.

    Returns:
        Fixed-size chunks.
    """
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size_chars, len(text))
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end

    return chunks

def _apply_overlap(chunks: list[str], overlap_chars: int) -> list[str]:
    """Add overlap by prefixing each chunk with previous chunk tail.

    Args:
        chunks: Base chunks without overlap.
        overlap_chars: Number of previous chars to prepend.

    Returns:
        Chunks with overlap applied.
    """
    if overlap_chars <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for chunk in chunks[1:]:
        prefix = overlapped[-1][-overlap_chars:]
        overlapped.append(prefix + chunk)
    return overlapped
