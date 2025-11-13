import pathlib
from langchain_community.document_loaders import TextLoader

def first_letter(title: str) -> str:
    for ch in title.strip():
        if ch.isalnum():
            return ch.upper()
    return "#"

def discover_paths(data_dir: pathlib.Path, allowed_ext=(".md", ".txt")) -> list[pathlib.Path]:
    """Find all files with given extensions under data_dir."""
    return [
        p for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in allowed_ext
    ]

# Only search entries whose first_letter is "A"
# retriever_A = vs.as_retriever(search_kwargs={"k": 5, "filter": {"first_letter": "A"}})
# docs = retriever_A.get_relevant_documents("what are adverse effects?")

def load_documents_from_paths(paths: list[pathlib.Path]) -> list:
    """Load text/markdown files into LangChain Document objects."""
    all_docs = []
    for p in paths:
        docs = TextLoader(str(p), encoding="utf-8").load()
        for d in docs:
            d.metadata = {
                **d.metadata,
                "source": str(p),
                "filename": p.name,
                "first_letter": first_letter(p.stem),
                "ext": p.suffix.lower(),
            }
        all_docs.extend(docs)
    return all_docs