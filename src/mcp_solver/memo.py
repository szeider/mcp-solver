import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class MemoManager:
    """File-backed text manager."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.content = ""
        self._ensure_file_exists()
        self._read_file()

    def _ensure_file_exists(self) -> None:
        """Create memo file and parent directories if they don't exist."""
        try:
            path = Path(self.filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.touch()
        except Exception as e:
            logger.error(f"Failed to create memo file: {e}")
            raise

    def _read_file(self) -> None:
        """Read content from memo file."""
        try:
            with open(self.filepath, 'r') as f:
                self.content = f.read()
        except Exception as e:
            logger.error(f"Failed to read memo file: {e}")
            raise

    def _write_file(self) -> None:
        """Write current content to memo file."""
        try:
            with open(self.filepath, 'w') as f:
                f.write(self.content)
        except Exception as e:
            logger.error(f"Failed to write memo file: {e}")
            raise

    def get_lines(self) -> list[str]:
        return self.content.splitlines(keepends=True) if self.content else []

    def edit_range(self, line_start: int, line_end: Optional[int], new_content: str) -> None:
        """Edit content within given line range (1-based indexing) and save to file."""
        lines = self.get_lines()
        line_start = max(1, line_start) - 1
        
        if not lines:  # Empty or new content
            self.content = new_content
            self._write_file()
            return

        line_end = len(lines) if line_end is None else min(line_end, len(lines))
        if line_start > len(lines):
            line_start = len(lines)
            
        if not new_content.endswith('\n'):
            new_content += '\n'
            
        new_lines = new_content.splitlines(keepends=True)
        lines[line_start:line_end] = new_lines
        self.content = ''.join(lines)
        self._write_file()