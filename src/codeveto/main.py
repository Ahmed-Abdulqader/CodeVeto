import sys
import pathlib
import sqlite3
import os
import json
from context_engine.chunks_saver import (
    init_database,
    save_file,
    resolve_parent_classes)
from context_engine.config import (
    DATABASE_PATH
)
from context_engine.chunk_query import (
    get_file_info
)   
from agents.tools import get_file_classes

def main(con: sqlite3.Connection) -> None:

    file = input("Choose the file path that you want to edit: ")
    file_path = pathlib.Path(file.strip()).resolve()
    if not (file_path.is_file() and file_path.suffix.lower() == '.py'):
        print("file not exsits or not found")
        return
    res = get_file_info(conn, file_path)
    if isinstance(res, str):
        res = json.loads(res)
    if 'error' in res:
        save_file(conn, file_path)
        resolve_parent_classes(conn)
    last_modify = os.path.getmtime(file_path)
    last_modify_database = res["last_modified"]
    if not (str(last_modify).strip() == str(last_modify_database).strip):
        save_file(conn, file_path)
        resolve_parent_classes(conn)
        
    # user_prompt = input("init CodeVeto!\nUser>>")
    out = get_file_classes(conn, file)
    print(out)

    



if __name__=="__main__":
    try:
        init_database(DATABASE_PATH)
        with sqlite3.connect(str(DATABASE_PATH)) as conn:
            main(conn)
    except Exception as e:
        print(f"error {e}")

