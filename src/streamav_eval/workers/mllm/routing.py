"""Metric-to-prompt, schema, duration, and modality routing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .schemas import (
    AQ_SCHEMA,
    ASR_SCHEMA,
    AUF_SCHEMA,
    HDF_SCHEMA,
    PAC_SCHEMA,
    PVC_SCHEMA,
    SHARED_INSTRUCTION_SCHEMA,
    VQ_SCHEMA,
    VSR_SCHEMA,
    VUF_SCHEMA,
)


class MLLMMetric(str, Enum):
    VQ = "VQ"
    AQ = "AQ"
    VIF = "VIF"
    AIF = "AIF"
    VID_EARLY = "VID-Early"
    VID_LATE = "VID-Late"
    AID_EARLY = "AID-Early"
    AID_LATE = "AID-Late"
    P0_VIF = "P0-VIF"
    P0_AIF = "P0-AIF"
    VUF = "VUF"
    AUF = "AUF"
    VSR = "VSR"
    ASR = "ASR"
    HDF_ADJ = "HDF-Adjacent"
    HDF_LR = "HDF-Long-Range"
    PVC = "PVC"
    PAC = "PAC"


class MediaMode(str, Enum):
    NONE = "none"
    VIDEO_ONLY = "video_only"
    SYNCHRONIZED_AV = "synchronized_av"
    AUDIO_ONLY = "audio_only"


FIXED_USER_PROMPT = "Evaluate the provided media according to the system instruction."
_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "resources" / "prompts"


@dataclass(frozen=True, slots=True)
class MLLMRoute:
    metric: MLLMMetric
    media_mode: MediaMode
    expected_duration_s: int | None
    system_prompt_file: str
    user_prompt_file: str | None
    response_schema: Mapping[str, Any]


_ROUTES = {
    MLLMMetric.VQ: MLLMRoute(
        MLLMMetric.VQ,
        MediaMode.VIDEO_ONLY,
        30,
        "vq_judge.txt",
        None,
        VQ_SCHEMA,
    ),
    MLLMMetric.AQ: MLLMRoute(
        MLLMMetric.AQ,
        MediaMode.SYNCHRONIZED_AV,
        30,
        "aq_judge.txt",
        None,
        AQ_SCHEMA,
    ),
    MLLMMetric.VIF: MLLMRoute(
        MLLMMetric.VIF,
        MediaMode.VIDEO_ONLY,
        180,
        "vif_criterion_judge.txt",
        None,
        SHARED_INSTRUCTION_SCHEMA,
    ),
    MLLMMetric.AIF: MLLMRoute(
        MLLMMetric.AIF,
        MediaMode.SYNCHRONIZED_AV,
        180,
        "aif_criterion_judge.txt",
        None,
        SHARED_INSTRUCTION_SCHEMA,
    ),
    MLLMMetric.VID_EARLY: MLLMRoute(
        MLLMMetric.VID_EARLY,
        MediaMode.VIDEO_ONLY,
        30,
        "vid_endpoint_judge.txt",
        None,
        SHARED_INSTRUCTION_SCHEMA,
    ),
    MLLMMetric.VID_LATE: MLLMRoute(
        MLLMMetric.VID_LATE,
        MediaMode.VIDEO_ONLY,
        30,
        "vid_endpoint_judge.txt",
        None,
        SHARED_INSTRUCTION_SCHEMA,
    ),
    MLLMMetric.AID_EARLY: MLLMRoute(
        MLLMMetric.AID_EARLY,
        MediaMode.SYNCHRONIZED_AV,
        30,
        "aid_endpoint_judge.txt",
        None,
        SHARED_INSTRUCTION_SCHEMA,
    ),
    MLLMMetric.AID_LATE: MLLMRoute(
        MLLMMetric.AID_LATE,
        MediaMode.SYNCHRONIZED_AV,
        30,
        "aid_endpoint_judge.txt",
        None,
        SHARED_INSTRUCTION_SCHEMA,
    ),
    MLLMMetric.P0_VIF: MLLMRoute(
        MLLMMetric.P0_VIF,
        MediaMode.VIDEO_ONLY,
        30,
        "vif_criterion_judge.txt",
        None,
        SHARED_INSTRUCTION_SCHEMA,
    ),
    MLLMMetric.P0_AIF: MLLMRoute(
        MLLMMetric.P0_AIF,
        MediaMode.SYNCHRONIZED_AV,
        30,
        "aif_criterion_judge.txt",
        None,
        SHARED_INSTRUCTION_SCHEMA,
    ),
    MLLMMetric.VUF: MLLMRoute(
        MLLMMetric.VUF,
        MediaMode.SYNCHRONIZED_AV,
        30,
        "vuf_judge.txt",
        None,
        VUF_SCHEMA,
    ),
    MLLMMetric.AUF: MLLMRoute(
        MLLMMetric.AUF,
        MediaMode.SYNCHRONIZED_AV,
        30,
        "auf_judge.txt",
        None,
        AUF_SCHEMA,
    ),
    MLLMMetric.VSR: MLLMRoute(
        MLLMMetric.VSR,
        MediaMode.VIDEO_ONLY,
        30,
        "vsr_judge.txt",
        None,
        VSR_SCHEMA,
    ),
    MLLMMetric.ASR: MLLMRoute(
        MLLMMetric.ASR,
        MediaMode.AUDIO_ONLY,
        30,
        "asr_judge.txt",
        None,
        ASR_SCHEMA,
    ),
    MLLMMetric.HDF_ADJ: MLLMRoute(
        MLLMMetric.HDF_ADJ,
        MediaMode.SYNCHRONIZED_AV,
        30,
        "hdf_judge.txt",
        None,
        HDF_SCHEMA,
    ),
    MLLMMetric.HDF_LR: MLLMRoute(
        MLLMMetric.HDF_LR,
        MediaMode.SYNCHRONIZED_AV,
        30,
        "hdf_judge.txt",
        None,
        HDF_SCHEMA,
    ),
    MLLMMetric.PVC: MLLMRoute(
        MLLMMetric.PVC,
        MediaMode.VIDEO_ONLY,
        4,
        "pvc_judge.txt",
        None,
        PVC_SCHEMA,
    ),
    MLLMMetric.PAC: MLLMRoute(
        MLLMMetric.PAC,
        MediaMode.AUDIO_ONLY,
        4,
        "pac_judge.txt",
        None,
        PAC_SCHEMA,
    ),
}


def route_for(metric: MLLMMetric | str) -> MLLMRoute:
    aliases = {
        "VQ": MLLMMetric.VQ,
        "AQ": MLLMMetric.AQ,
        "VIF": MLLMMetric.VIF,
        "AIF": MLLMMetric.AIF,
        "P0-VIF": MLLMMetric.P0_VIF,
        "P0-AIF": MLLMMetric.P0_AIF,
        "VID-Early": MLLMMetric.VID_EARLY,
        "VID-Late": MLLMMetric.VID_LATE,
        "AID-Early": MLLMMetric.AID_EARLY,
        "AID-Late": MLLMMetric.AID_LATE,
        "VUF": MLLMMetric.VUF,
        "AUF": MLLMMetric.AUF,
        "VSR": MLLMMetric.VSR,
        "ASR": MLLMMetric.ASR,
        "HDF-Adjacent": MLLMMetric.HDF_ADJ,
        "HDF-Long-Range": MLLMMetric.HDF_LR,
        "PVC": MLLMMetric.PVC,
        "PAC": MLLMMetric.PAC,
    }
    try:
        if isinstance(metric, MLLMMetric):
            parsed = metric
        elif metric in aliases:
            parsed = aliases[metric]
        else:
            parsed = MLLMMetric(metric)
    except ValueError as exc:
        raise ValueError(f"unsupported MLLM metric: {metric!r}") from exc
    return _ROUTES[parsed]


def route_protocol_sha256(
    metric: MLLMMetric | str,
    *,
    prompts_dir: str | Path | None = None,
) -> str:
    """Hash every frozen input that determines an MLLM route's judgment."""

    route = route_for(metric)
    root = Path(prompts_dir) if prompts_dir is not None else _PROMPTS_DIR
    prompt = (root / route.system_prompt_file).read_text(encoding="utf-8")
    payload = {
        "metric": route.metric.value,
        "media_mode": route.media_mode.value,
        "expected_duration_s": route.expected_duration_s,
        "system_prompt_file": route.system_prompt_file,
        "system_prompt": prompt,
        "user_prompt": FIXED_USER_PROMPT,
        "response_schema": route.response_schema,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
