from typing import Any

from pydantic import BaseModel


class FileMetadata(BaseModel):
    name: str
    path: str
    modified_ts: float


class Chunk(BaseModel):
    id: str
    language: str
    chunk_type: str
    docstring: str | None
    signature: str | None
    content: str
    start_line: int
    end_line: int
    file_metadata: FileMetadata

    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "language": self.language,
            "chunk_type": self.chunk_type,
            "docstring": self.docstring,
            "signature": self.signature,
            "content": self.content,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "file_metadata": self.file_metadata,
        }
        return d
