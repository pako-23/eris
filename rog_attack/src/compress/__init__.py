from .compressor import *

compress_registry = {
    "uniform": UniformQuantizer,
    "topk": Topk,
    "qsgd": QsgdQuantizer,
    "DPSGD": DPSGDCompressor,
    "pruning": PruneLargest,
    "eris": ErisCompressor,
    "randomcompress": RandomCompressor,
    "eris_partial": ErisPartialCompressor,  
}
