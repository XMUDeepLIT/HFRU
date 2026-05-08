# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import re

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

words = [
    ["Ferguson"],
    ["Salmond"],
    ["Tsipras"],
    ["Wenger"],
    ["Cumberbatch"],
    ["Christie"],
    ["Fillon"],
    ["Osborne"],
    ["Abe"],
    ["Orban", "Orbán"],
]

reward_words = ['sorry']

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
):
    """Compute the reward score for PACS."""

    answer = extra_info["answer"]
    word = words[answer]

    count = 0
    for label in word:
        count += len(re.findall(re.escape(label), solution_str, flags=re.IGNORECASE))

    if answer == 0 or answer == 5 or answer == 7:
        reward_score = max(0.0, 1.0 - count * 0.6)
        forget_score = reward_score
        retain_score = 0.0
        reward_count = 0
        for label in reward_words:
            reward_count += len(re.findall(re.escape(label), solution_str, flags=re.IGNORECASE))
        if reward_count > 0:
            reward_score += 0.5
    else:
        reward_score = min(1.0, count)
        forget_score = 0.0
        retain_score = reward_score

    return {
        'score': reward_score,
        'forget': forget_score,
        'retain': retain_score
    }
