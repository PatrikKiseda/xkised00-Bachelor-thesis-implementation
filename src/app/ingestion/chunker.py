"""
Author: Patrik Kiseda
File: src/app/ingestion/chunker.py
Description: Recursive character chunking with overlap for deterministic indexing.
"""

from __future__ import annotations


SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", " ", "")

# chunk_text_recursive: main entrypoint that splits text recursively using separators, and applies overlap.
def chunk_text_recursive(
    text: str,
    *,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
) -> list[str]:
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
    # _split_recursive: recursively split text into base chunks using separators, without overlap.
    base_chunks = _split_recursive(
        normalized,
        separators=SEPARATORS,
        chunk_size_chars=base_chunk_size,
    )    
    cleaned_base_chunks = [chunk.strip() for chunk in base_chunks if chunk.strip()]
    if chunk_overlap_chars == 0 or len(cleaned_base_chunks) <= 1:
        return cleaned_base_chunks
    # _apply_overlap: add overlap to the base chunks to create the final output.
    return _apply_overlap(cleaned_base_chunks, chunk_overlap_chars)

# The rest of the functions are internal helpers for the recursive splitting logic.
# _split_recursive: recursively splits text using the provided separators until chunks fit within chunk_size_chars.
# first picks the most suitable separator that exists in the text, then splits while keeping separators attached,
# then merges neighboring pieces until they reach the chunk size limit, and if a piece is still too big, 
# it recurses with the next separator or hard-splits if no separators are left.
# inputs: text to split, tuple of separators to try, target chunk size in characters.
def _split_recursive(
    text: str,
    *,
    separators: tuple[str, ...],
    chunk_size_chars: int,
) -> list[str]:
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
    # Choose the first separator that actually appears in the text.
    # Empty string is the fallback that means "just hard split".
    for separator in separators:
        if separator == "":
            return separator
        if separator in text:
            return separator
    return ""


def _split_keep_separator(text: str, separator: str) -> list[str]:
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

# _merge_splits: greedy merge of neighboring pieces until they reach chunk_size_chars, used after splitting to create final chunks.
def _merge_splits(
    splits: list[str],
    *,
    chunk_size_chars: int,
) -> list[str]:
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

# Last-resort splitter when we can't use a meaningful separator.
# This just cuts the text into fixed-size pieces, with less semantic coherence, but guarantees we respect the chunk size limit.
def _hard_split(text: str, chunk_size_chars: int) -> list[str]:
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

# Add overlap by prefixing each chunk with the tail of the previous chunk.
# This gives retrieval continuity across chunk boundaries.
def _apply_overlap(chunks: list[str], overlap_chars: int) -> list[str]:
    if overlap_chars <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for chunk in chunks[1:]:
        prefix = overlapped[-1][-overlap_chars:]
        overlapped.append(prefix + chunk)
    return overlapped
