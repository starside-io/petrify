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

"""Local web page for ``petrify run --web``.

:mod:`petrify.server.control` (the loop-to-server hand-off) has no web
dependencies. :mod:`petrify.server.app` imports FastAPI and uvicorn only when a
server is created; they come with the optional extra ``petrify[web]``.
"""

from __future__ import annotations

from petrify.server.control import RUNTIME_FIELDS, RunControl, RunSnapshot

__all__ = ["RUNTIME_FIELDS", "RunControl", "RunSnapshot"]
