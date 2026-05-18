#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a controlled QA benchmark from eval_strategy_specs.json.

The specs use stable anchors inside data/eval_*.md. This script resolves each
anchor to chunk identities under the current parent-child chunking config, so
the resulting JSON can be used directly by scripts/eval.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).parent.parent))

from RAG.core.config import RAGConfig
from RAG.utils import ReadFiles


def _basename(path: str) -> str:
    return os.path.basename(path.replace("\\", "/"))


def _load_specs(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("spec file must contain a JSON list")
    return data


def _chunk_data(data_dir: str, config: RAGConfig):
    pc = config.parent_child
    reader = ReadFiles(data_dir)
    return reader.get_parent_child_documents(
        child_max_tokens=pc.child_max_tokens,
        child_overlap_tokens=pc.child_overlap_tokens,
        parent_max_tokens=pc.parent_max_tokens,
    )


def _resolve_relevant_ids(
    spec: dict[str, Any],
    docs,
    parent_map: dict[str, str],
    max_ids: int,
) -> tuple[list[str], list[str]]:
    source = spec["source"]
    anchor = spec["anchor"]
    relevant_ids: list[str] = []
    contexts: list[str] = []

    for doc in docs:
        if _basename(doc.metadata.get("source", "")) != source:
            continue
        parent_text = parent_map.get(doc.metadata.get("parent_id"), "")
        haystack = f"{doc.text}\n{parent_text}"
        if anchor not in haystack:
            continue
        relevant_ids.append(doc.identity)
        context = parent_text or doc.text
        if context and context not in contexts:
            contexts.append(context[:1200])
        if len(relevant_ids) >= max_ids:
            break

    return relevant_ids, contexts


def build(args) -> dict[str, Any]:
    specs = _load_specs(args.specs)
    config = RAGConfig()
    docs, parent_map = _chunk_data(args.data_dir, config)
    items: list[dict[str, Any]] = []
    unresolved: list[str] = []

    for spec in specs:
        relevant_ids, contexts = _resolve_relevant_ids(
            spec,
            docs,
            parent_map,
            max_ids=args.max_relevant_ids,
        )
        if not relevant_ids:
            unresolved.append(spec.get("id", spec["question"]))
            if args.strict:
                continue

        items.append({
            "id": spec.get("id"),
            "question": spec["question"],
            "ground_truth": spec["ground_truth"],
            "qa_type": spec.get("qa_type", "concept"),
            "source": os.path.join(args.data_dir, spec["source"]),
            "section": spec.get("anchor"),
            "retrieval_focus": spec.get("retrieval_focus"),
            "relevant_contexts": contexts,
            "relevant_ids": relevant_ids,
        })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    report = {
        "spec_count": len(specs),
        "output_count": len(items),
        "unresolved": unresolved,
        "avg_relevant_ids": (
            round(sum(len(item["relevant_ids"]) for item in items) / len(items), 3)
            if items else 0
        ),
        "output": args.output,
    }
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build controlled TinyRAG QA benchmark")
    parser.add_argument("--specs", default="benchmark/eval_strategy_specs.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="benchmark/qa_strategy_controlled.json")
    parser.add_argument("--report", default="benchmark/qa_strategy_controlled_report.json")
    parser.add_argument("--max-relevant-ids", type=int, default=4)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

