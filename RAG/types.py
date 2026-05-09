#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Protocol


class EmbeddingModel(Protocol):
    def get_embedding(self, text: str) -> List[float]:
        ...
