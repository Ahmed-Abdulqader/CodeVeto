import json
import pathlib
import tiktoken
from pydantic_ai import RunContext
from typing import List

from models.agents_models import DatabaseDeps

from context_engine.chunk_query import (
    get_imports,
    get_top_level_chunks,
    get_classes,
    get_methods,
    get_functions
)


# add comments and docstrings


def _fewer_tokens(chunk, code_only):
    proxy_encoder = tiktoken.get_encoding("o200k_base")
    chunk1_tokens = len(proxy_encoder.encode(chunk))
    code_tokens = len(proxy_encoder.encode(code_only))
    if chunk1_tokens <= code_tokens:
        return chunk
    else:
        code = f"""
```python
f{code_only}
```
"""
        return code


def _file_path_wrapper(file_name:str)->str:
    file_path = pathlib.Path(file_name.strip()).resolve()
    if not (file_path.is_file() and file_path.suffix.lower() == '.py'):
        return "not found"
    return file_path
    
# when changed add send only code of the imports
def get_file_imports(ctx: RunContext[DatabaseDeps], file_name: str) -> str | List[dict]:
    file_path = _file_path_wrapper(file_name)
    if file_path == "not found" :
        return f"file {file_name} wasn't found"
    imports = get_imports(ctx.deps.db, file_path)
    imports = json.loads(imports)

    return imports

# the top level chunks should be return only code
def get_file_top_level_chunk(ctx: RunContext[DatabaseDeps], file_name:str) -> str | List[dict]:
    file_path = _file_path_wrapper(file_name)
    if file_path == "not found" :
        return f"file {file_name} wasn't found"
    top_level_chunks = get_top_level_chunks(ctx.deps.db, file_path)
    top_level_chunks = json.loads(top_level_chunks)
    
    excluded_keys = {"id", "file_id"}
    return [
        {k: v for k,v in chunk.items() if k not in excluded_keys}
        for chunk in top_level_chunks
    ]

def get_file_classes(ctx: RunContext[DatabaseDeps], file_name:str) -> str | List[dict]:
    file_path = _file_path_wrapper(file_name)
    if file_path == "not found" :
        return f"file {file_name} wasn't found"
    classes = get_classes(ctx.deps.db, file_path)
    classes = json.loads(classes)
    final_clses = []
    for cls in classes:
        tmp_cls = cls.copy()
        methods = get_methods(ctx.deps.db, cls['id'])
        methods = json.loads(methods)
        methods_names = [method['name'] for method in methods]
        tmp_cls.pop("code", None)
        tmp_cls["methods"] = methods_names
        efficient_cls = _fewer_tokens(str(tmp_cls), cls['code'])
        final_clses.append(efficient_cls)

    return final_clses

def get_class_methods_signature():
    pass

def get_file_functions():
    pass

def expand_function(function_name):
    pass

def expand_class(cls_name):
    pass

def expand_method(cls_name, method_name):
    pass

def write_code_chunk(file_name, start_line, end_line, start_column, end_column):
    pass


