# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

ExpertMapping = tuple[str, str, int, str]


def load_rank_sliced_expert_weight(
    name: str,
    weight: Any,
    params: Mapping[str, Any],
    mappings: Sequence[ExpertMapping],
) -> str | None:
    for param_name, weight_name, expert_id, shard_id in mappings:
        if weight_name not in name:
            continue
        mapped_name = name.replace(weight_name, param_name)
        param = params.get(mapped_name)
        if param is None:
            continue
        success = param.weight_loader(
            param,
            weight,
            mapped_name,
            shard_id=shard_id,
            expert_id=expert_id,
            return_success=True,
        )
        if success:
            return mapped_name
    return None
