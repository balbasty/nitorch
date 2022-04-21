import inspect


def get_closure_ls(closure):
    if 'in_line_search' in inspect.signature(closure).parameters:
        return lambda *a, **k: closure(*a, **k, in_line_search=True)
    return closure
