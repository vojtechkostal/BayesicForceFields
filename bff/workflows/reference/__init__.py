def main(*args, **kwargs):
    from .main import main as _main
    return _main(*args, **kwargs)


def run_job(*args, **kwargs):
    from .main import run_job as _run_job
    return _run_job(*args, **kwargs)


__all__ = ['main', 'run_job']
