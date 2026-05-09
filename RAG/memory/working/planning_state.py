#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PlanningState — multi-step task decomposition.
Useful when the RAG system is used as an agent that works
toward a goal over multiple queries (e.g. ReflectionEngine retries).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PlanningState:
    """
    Optional goal-tracking state.
    Only populated when the caller sets a goal explicitly.
    """
    current_goal: Optional[str] = None
    sub_goals: List[str] = field(default_factory=list)
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)

    def set_goal(self, goal: str) -> None:
        self.current_goal = goal

    def add_sub_goal(self, sub_goal: str) -> None:
        if sub_goal not in self.sub_goals:
            self.sub_goals.append(sub_goal)

    def complete_step(self, step: str) -> None:
        if step not in self.completed_steps:
            self.completed_steps.append(step)

    def fail_step(self, step: str) -> None:
        if step not in self.failed_steps:
            self.failed_steps.append(step)

    @property
    def is_complete(self) -> bool:
        return bool(self.current_goal) and set(self.sub_goals) <= set(self.completed_steps)

    def to_str(self) -> str:
        if not self.current_goal:
            return ""
        lines = [f"Goal: {self.current_goal}"]
        for sg in self.sub_goals:
            status = "✓" if sg in self.completed_steps else ("✗" if sg in self.failed_steps else "○")
            lines.append(f"  [{status}] {sg}")
        return "\n".join(lines)
