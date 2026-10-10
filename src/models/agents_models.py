from dataclasses import dataclass
from sqlite3 import Connection


@dataclass
class DatabaseDeps:
    db: Connection
