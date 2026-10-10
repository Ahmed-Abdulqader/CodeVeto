# constants
import pathlib


def _database():
    database = pathlib.Path().cwd() / ".codeveto" / "codeveto.db"
    database.parent.mkdir(parents=True, exist_ok=True)
    database.touch(exist_ok=True)
    return database


HOME_DIR = pathlib.Path.home() / ".codeveto"
DATABASE_PATH = _database()

FUNC_COLS = (
    "type, name, docstring, returns, start_line, end_line, "
    "start_column, end_column, args, calls, decorators, code"
)
FUNC_VALS = (
    ":type, :name, :docstring, :returns, :start_line, :end_line, "
    ":start_column, :end_column, :args, :calls, :decorators, :code"
)


EMBED_ONNX = HOME_DIR / "models/jina-embeddings-v2-base-code/model_quantized.onnx"
TOKENIZER = HOME_DIR / "models/jina-embeddings-v2-base-code/tokenizer.json"
INDEX_DB = pathlib.Path(".codeveto") / "code_index.db"

print(HOME_DIR)
