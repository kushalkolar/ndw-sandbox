from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

from rendercanvas.utils.asyncs import sleep as async_sleep
from cachetools import LFUCache, LRUCache
from PyNvVideoCodec import OutputColorType, SimpleDecoder
import numpy as np
import torch
import fastplotlib as fpl

from copy import copy

class NvidiaReader:
    def __init__(self, path, cache_size: int = 32):
        # Create our decoder instance
        self.decoder = SimpleDecoder(
            enc_file_path=path,
            gpu_id=0,
            use_device_memory=True,
            output_color_type=OutputColorType.RGB,
            need_scanned_stream_metadata=True,
        )
        
        self._cache = LRUCache(maxsize=cache_size)

        # used for pre-fetching when idle
        self._last_index = 0
    
        self._metadata = self.decoder.get_scanned_stream_metadata()
        self.key_frames = np.asarray(self._metadata.key_frame_indices)
        self._cache_size = cache_size
    
    def __array__(self, dtype=None, copy=None):
        if copy:
            return copy(self)
            
        return self

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise NotImplementedError

    def __array_function__(self, func, types, *args, **kwargs):
        raise NotImplementedError

    @property
    def dtype(self) -> str:
        return np.uint8

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self._metadata.num_frames, self._metadata.height, self._metadata.width, 3)

    @property
    def min(self) -> float:
        return 0.0

    @property
    def max(self) -> float:
        return 255.0

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def nbytes(self) -> int:
        return np.prod(self.shape + (np.dtype(self.dtype).itemsize,), dtype=np.int64)

    def _decode_frame(self, index: int):
        frame = torch.from_dlpack(self.decoder.get_batch_frames_by_index([index])[0]).cpu().numpy()
        self._cache[index] = frame
        return frame

    def _decode_batch(self, start_index: int):
        frames = self.decoder.get_batch_frames_by_index(list(range(start_index, start_index + self._cache_size)))
        frames_t = torch.stack([torch.from_dlpack(f) for f in frames])
        frames_t = frames_t.cpu().numpy()

        for i in range(start_index, start_index + self._batch_size):
            self._cache[i] = frames_t

    def _decode_when_idle(self):
        # while True:
            # fill the cache based on the most recent index
            window = self._cache_size // 2

            # get a window of frames around the most recent index
            start = max(min(self._last_index - window, self.shape[0]), 0)
            stop = max(min(self._last_index + window, self.shape[0]), 0)
            
            cached_frames = set(self._cache.keys())
            
            to_fetch = sorted(set(range(start, stop)) - cached_frames)
            frames = self.decoder.get_batch_frames_by_index(to_fetch)
            frames_t = torch.stack([torch.from_dlpack(f) for f in frames])
            frames_t = frames_t.cpu().numpy()

            for i, frame in zip(to_fetch, frames_t):
                self._cache[i] = frame

            # await async_sleep(0.01)

    def __getitem__(self, indices: tuple[slice, ...]) -> np.ndarray:
        # indices can be a tuple of slice | Ellipsis
        # need to accoutn for Ellipsis as the last object in the tuple
        index = indices[0].start
        self._last_index = index
        
        if index in self._cache.keys():
            return self._cache[index][None]
        else:
            return self._decode_frame(index)[None]
