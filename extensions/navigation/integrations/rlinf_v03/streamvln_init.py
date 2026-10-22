__all__ = ["StreamVLNEnv"]


def __getattr__(name: str):
    if name != "StreamVLNEnv":
        raise AttributeError(name)
    from .streamvln_env import StreamVLNEnv

    globals()[name] = StreamVLNEnv
    return StreamVLNEnv
