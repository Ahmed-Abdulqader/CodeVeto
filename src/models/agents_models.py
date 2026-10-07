from sqlite3 import Connection
from dataclasses import dataclass

@dataclass
class DatabaseDeps:
    db: Connection

