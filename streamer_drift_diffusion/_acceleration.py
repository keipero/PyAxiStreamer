"""Numbaは任意。無効時にも同じ走査順のPython実装を使用する。"""
import os

NUMBA_AVAILABLE = False
if os.environ.get("STREAMER_DISABLE_NUMBA") != "1":
    try:
        from numba import njit, prange, set_num_threads, get_num_threads
        NUMBA_AVAILABLE = True
    except ImportError:
        pass

if not NUMBA_AVAILABLE:
    prange = range

    def set_num_threads(count):
        """Numba無効時は、指定にかかわらず逐次実行する。"""

    def get_num_threads():
        return 1

    def njit(*args, **kwargs):
        def identity(function):
            return function
        return identity
