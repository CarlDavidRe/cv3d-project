"""Dataset interfaces for next-best-view experiments."""

from nbv.data.num_dataset import (
    NUMDataset,
    NUMDatasetError,
    NUMSample,
    NUMSampleRecord,
    load_object_split,
    rgb_to_float_chw,
)
from nbv.data.num_splits import (
    PUN_HELD_OUT_CATEGORIES,
    build_num_v1_manifest,
    complete_objects_by_category,
    pun_compatible_object_split,
)
from nbv.data.visibility_cache import (
    VISIBILITY_CACHE_SCHEMA_VERSION,
    IncompatibleVisibilityCacheError,
    VisibilityCache,
    VisibilityCacheError,
    load_visibility_cache,
    save_visibility_cache,
    visibility_cache_compatibility_errors,
    visibility_cache_path,
)

__all__ = [
    "NUMDataset",
    "NUMDatasetError",
    "NUMSample",
    "NUMSampleRecord",
    "PUN_HELD_OUT_CATEGORIES",
    "build_num_v1_manifest",
    "complete_objects_by_category",
    "load_object_split",
    "pun_compatible_object_split",
    "rgb_to_float_chw",
    "VISIBILITY_CACHE_SCHEMA_VERSION",
    "IncompatibleVisibilityCacheError",
    "VisibilityCache",
    "VisibilityCacheError",
    "load_visibility_cache",
    "save_visibility_cache",
    "visibility_cache_compatibility_errors",
    "visibility_cache_path",
]
