#!/usr/bin/env python3
"""Minimal credential-safe live preflight for the U7 Polaris web tools."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

from unirl.rollout.loop.tools.search import SearchTool
from unirl.rollout.loop.tools.visit import VisitTool


def _assert_healthy(name: str, diagnostics: Mapping[str, Any]) -> None:
    if int(diagnostics.get("success_count", 0)) < 1:
        raise RuntimeError(f"{name} preflight produced no successful request")
    for field in (
        "transient_exhausted_count",
        "permanent_error_count",
        "auth_error_count",
    ):
        if int(diagnostics.get(field, 0)) != 0:
            raise RuntimeError(f"{name} preflight reported {field}")


def main() -> None:
    search = SearchTool()
    search_live = search.execute_with_info({"query": "apple inc"})
    _assert_healthy("search", search_live.diagnostics)
    search_cached = search.execute_with_info({"query": "apple inc"})
    if int(search_cached.diagnostics.get("cache_hit_count", 0)) != 1:
        raise RuntimeError("search preflight did not exercise the success cache")

    # The preflight validates Jina itself, not the separately served summarizer.
    summary_url = os.environ.pop("SUMMARY_URL", None)
    try:
        visit = VisitTool(reader_provider="jina_ai")
        visit_live = visit.execute_with_info(
            {"url": "https://www.google.com", "goal": "confirm readable page content"}
        )
        _assert_healthy("visit", visit_live.diagnostics)
        visit_cached = visit.execute_with_info(
            {"url": "https://www.google.com#u7-cache", "goal": "confirm cache reuse"}
        )
        if int(visit_cached.diagnostics.get("cache_hit_count", 0)) != 1:
            raise RuntimeError("visit preflight did not exercise the success cache")
    finally:
        if summary_url is not None:
            os.environ["SUMMARY_URL"] = summary_url

    # Diagnostics are allow-listed aggregates only; never print result text,
    # queries, URLs, headers, response bodies, or credential-bearing exceptions.
    print(
        json.dumps(
            {
                "search": dict(search_live.diagnostics),
                "search_cache": dict(search_cached.diagnostics),
                "visit": dict(visit_live.diagnostics),
                "visit_cache": dict(visit_cached.diagnostics),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
