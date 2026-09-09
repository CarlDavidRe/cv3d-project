"""Trainable prediction modules for next-best-view experiments."""

from nbv.models.heads import (
    FixedMapHead,
    LightweightProbeHead,
    count_trainable_parameters,
)
from nbv.models.independent_multiview import (
    IndependentHistoryGainModel,
    independent_feature_batch,
)
from nbv.models.joint_multiview import JointHistoryGainModel
from nbv.models.pun import (
    PUN_CHECKPOINT_FILE_ID,
    PUN_CHECKPOINT_SHA256,
    PUN_CHECKPOINT_URL,
    PUN_REFERENCE_COMMIT,
    PUN_RELEASE_NAME,
    PUN_REPOSITORY,
    PUNModelError,
    PUNUPNet,
    create_pun_transform,
    ensure_pun_checkpoint,
    load_pun_checkpoint,
    resolve_pun_data_config,
)

__all__ = [
    "FixedMapHead",
    "LightweightProbeHead",
    "IndependentHistoryGainModel",
    "JointHistoryGainModel",
    "PUNModelError",
    "PUNUPNet",
    "PUN_CHECKPOINT_FILE_ID",
    "PUN_CHECKPOINT_SHA256",
    "PUN_CHECKPOINT_URL",
    "PUN_REFERENCE_COMMIT",
    "PUN_RELEASE_NAME",
    "PUN_REPOSITORY",
    "count_trainable_parameters",
    "independent_feature_batch",
    "create_pun_transform",
    "ensure_pun_checkpoint",
    "load_pun_checkpoint",
    "resolve_pun_data_config",
]
