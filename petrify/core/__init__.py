# Copyright 2026 petrify contributors
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

# SPDX-License-Identifier: Apache-2.0

"""Core types for the petrify streaming engine."""

from __future__ import annotations

from petrify.core.config import (
    BackgroundConfig,
    BackgroundDiffConfig,
    EncodeConfig,
    EngineConfig,
    MotionConfig,
    PersonHogConfig,
    PetrifyConfig,
    RoiConfig,
)
from petrify.core.detections import Box, DetectionKind, Detections, Face
from petrify.core.engine import Engine, EngineStats, FrameTrace, Output
from petrify.core.frame import Frame, FrameMeta, SourceInfo
from petrify.core.plugin import Detector, Mode, Sink, Source, Stage

__all__ = [
    "BackgroundConfig",
    "BackgroundDiffConfig",
    "Box",
    "DetectionKind",
    "Detections",
    "Detector",
    "EncodeConfig",
    "Engine",
    "EngineConfig",
    "EngineStats",
    "Face",
    "Frame",
    "FrameMeta",
    "FrameTrace",
    "Mode",
    "MotionConfig",
    "Output",
    "PersonHogConfig",
    "PetrifyConfig",
    "RoiConfig",
    "Sink",
    "Source",
    "SourceInfo",
    "Stage",
]
