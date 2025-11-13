"""
split_text.py
---------------------------------
Utilities for splitting documents into chunks
for embedding and vector storage.
"""

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)
from typing import List
from langchain_core.documents import Document


def recursive_split(
    docs: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 200,
    length_function=len
) -> List[Document]:
    """
    Split documents into smaller overlapping chunks using a recursive splitter.
    Works well for Markdown, text, and structured content.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
    )
    return splitter.split_documents(docs)


def simple_split(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 0,
    separator: str = "\n"
) -> List[Document]:
    """
    A simpler non-recursive text splitter, useful for clean plain text.
    """
    splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)
