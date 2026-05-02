from datetime import datetime
from pathlib import Path

from rich.console import Console


class Logger:
    """Logs timestamped agent events to the console and a file."""

    LEVEL_STYLES = {
        "INFO": "white",
        "DECISION": "yellow",
        "EVOLUTION": "cyan",
        "RESULT": "green",
    }

    def __init__(self, log_file: str = "agent_log.txt") -> None:
        self.console = Console()
        self.log_file = Path(log_file)

    def log(self, level: str, message: str) -> None:
        level = level.upper()

        if level not in self.LEVEL_STYLES:
            supported = ", ".join(self.LEVEL_STYLES)
            raise ValueError(f"Unsupported log level '{level}'. Use one of: {supported}")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"

        self.console.print(entry, style=self.LEVEL_STYLES[level])

        with self.log_file.open("a", encoding="utf-8") as file:
            file.write(entry + "\n")

    def info(self, message: str) -> None:
        self.log("INFO", message)

    def decision(self, message: str) -> None:
        self.log("DECISION", message)

    def evolution(self, message: str) -> None:
        self.log("EVOLUTION", message)

    def result(self, message: str) -> None:
        self.log("RESULT", message)
