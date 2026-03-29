#!/usr/bin/env python3
import json
import subprocess
import sys
from argparse import ArgumentParser
from typing import Iterable

from cpu_perf_test import run_perf_test
 

COUNTER_TYPES = {
    "memory": ("uncore_imc", "uncore memory", "imc", "umc", "dram", "mem"),
    "cache": ("cache", "llc", "l1", "l2", "l3", "tlb"),
}


def load_perf_events() -> list[dict]:
    """Return the parsed JSON output of `perf list --no-desc -j`."""
    try:
        result = subprocess.run(
            ["perf", "list", "--no-desc", "-j"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        print("`perf` is not available in PATH. Install perf to query counters.", file=sys.stderr)
        sys.exit(2)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "no stderr from perf"
        print(f"`perf list --no-desc -j` failed with exit code {exc.returncode}: {stderr}", file=sys.stderr)
        sys.exit(exc.returncode)

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(f"Could not parse JSON returned by `perf list`: {exc}", file=sys.stderr)
        sys.exit(3)


def perf_counters_filter(terms: Iterable[str], counter_type: str) -> set[str]:
    """Return unique counters matching the requested search terms and counter family."""
    normalized_terms = [term.casefold() for term in terms if term.strip()]
    type_terms = COUNTER_TYPES[counter_type]
    matches: set[str] = set()

    for event in load_perf_events():
        searchable_fields = [
            str(event.get("Unit", "")),
            str(event.get("Topic", "")),
            str(event.get("EventName", "")),
            str(event.get("EventAlias", "")),
            str(event.get("BriefDescription", "")),
            str(event.get("PublicDescription", "")),
        ]
        haystack = " ".join(searchable_fields).casefold()

        if not any(type_term in haystack for type_term in type_terms):
            continue

        if normalized_terms and not any(term in haystack for term in normalized_terms):
            continue

        event_alias = str(event.get("EventAlias", "")).strip()
        event_name = str(event.get("EventName", "")).strip()

        if event_alias:
            matches.add(event_alias)
        elif event_name:
            matches.add(event_name)

    return matches


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Query `perf list` for memory-controller or cache-related counters."
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Space-separated search terms used to filter available perf counters.",
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=sorted(COUNTER_TYPES),
        help="Counter family to search: memory or cache.",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run `perf stat` for the matched counters against `sleep 5` after confirmation.",
    )

    args = parser.parse_args()
    matches = sorted(perf_counters_filter(args.query.split(), args.type))

    if not matches:
        print(
            f"No {args.type} counters matched query: {args.query!r}. "
            "Try broader terms or inspect `perf list --no-desc -j` directly.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\n".join(matches))
    sys.stdout.flush()

    if args.run_tests:
        sys.exit(run_perf_test(matches))
