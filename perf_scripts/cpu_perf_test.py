#!/usr/bin/env python3
from argparse import ArgumentParser
from collections import OrderedDict
import shlex
import subprocess
import sys
from typing import Sequence


def sanitize_event_for_test(event: str) -> str | None:
    """Return a test-safe event string, or None for meta/raw descriptors."""
    stripped = event.strip()
    if not stripped:
        return None

    if "0..255" in stripped or "0..15" in stripped:
        return None

    if ",edge" in stripped or ",umask" in stripped:
        return None

    return stripped


def sanitize_events_for_test(events: Sequence[str]) -> list[str]:
    """Prepare the event list for `perf stat` by removing unsupported range descriptors."""
    sanitized_events: list[str] = []
    for event in events:
        sanitized = sanitize_event_for_test(event)
        if sanitized:
            sanitized_events.append(sanitized)
    return sanitized_events


def build_perf_stat_command(events: Sequence[str], duration_seconds: int = 5) -> list[str]:
    """Build a `perf stat` command that measures all requested events."""
    command = ["perf", "stat"]
    for event in events:
        command.extend(["-e", event])
    command.extend(["sleep", str(duration_seconds)])
    return command


def build_sysctl_command(key: str, value: str) -> list[str]:
    """Build a sudo sysctl command used to prepare perf access."""
    return ["sudo", "sysctl", "-w", f"{key}={value}"]


def prepare_perf_environment() -> int:
    """Set sysctl knobs required for broader perf access before running tests."""
    commands = [
        build_sysctl_command("kernel.perf_event_paranoid", "-1"),
        build_sysctl_command("kernel.nmi_watchdog", "0"),
    ]

    for command in commands:
        print(f"Preparing perf environment: {shlex.join(command)}", file=sys.stderr)
        completed = subprocess.run(command)
        if completed.returncode != 0:
            return completed.returncode

    return 0


def group_key_for_event(event: str) -> str:
    """Return the leading group token for an event name."""
    for separator in (".", "/", ":"):
        if separator in event:
            return event.split(separator, 1)[0]
    underscore_parts = event.split("_")
    if len(underscore_parts) >= 3:
        return "_".join(underscore_parts[:2])
    if len(underscore_parts) >= 2:
        return underscore_parts[0]
    return event


def group_events_for_test(events: Sequence[str]) -> list[tuple[str, list[str]]]:
    """Group events by their leading name segment while preserving order."""
    grouped: OrderedDict[str, list[str]] = OrderedDict()
    for event in events:
        key = group_key_for_event(event)
        grouped.setdefault(key, []).append(event)
    return list(grouped.items())


def confirm_perf_test(event_groups: Sequence[tuple[str, Sequence[str]]], duration_seconds: int = 5) -> bool:
    """Show the test plan and let the user cancel before execution."""
    total_events = sum(len(events) for _, events in event_groups)
    print(
        f"About to run a perf smoke test for {total_events} event(s) in {len(event_groups)} batch(es) against `sleep {duration_seconds}`.",
        file=sys.stderr,
    )
    print(
        "This will first run `sudo sysctl -w kernel.perf_event_paranoid=-1` and "
        "`sudo sysctl -w kernel.nmi_watchdog=0`.",
        file=sys.stderr,
    )
    print("The following batches will be passed to `perf stat`:", file=sys.stderr)
    for group_name, events in event_groups:
        print(f"Group `{group_name}`:", file=sys.stderr)
        for event in events:
            print(f"  - {event}", file=sys.stderr)
        command_preview = shlex.join(build_perf_stat_command(events, duration_seconds))
        print(f"Command: {command_preview}", file=sys.stderr)

    try:
        answer = input("Proceed with the perf test? [y/N]: ").strip().casefold()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def run_perf_test(events: Sequence[str], duration_seconds: int = 5) -> int:
    """Run `perf stat` over `sleep` after user confirmation."""
    sanitized_events = sanitize_events_for_test(events)

    if not sanitized_events:
        print("No events were selected for the perf test.", file=sys.stderr)
        return 1

    event_groups = group_events_for_test(sanitized_events)

    if not confirm_perf_test(event_groups, duration_seconds):
        print("Perf test cancelled by user.", file=sys.stderr)
        return 130

    preparation_status = prepare_perf_environment()
    if preparation_status != 0:
        print("Failed to prepare perf environment. Aborting perf test.", file=sys.stderr)
        return preparation_status

    return_code = 0
    for group_name, group_events in event_groups:
        print(f"Running perf test batch `{group_name}`...", file=sys.stderr)
        command = build_perf_stat_command(group_events, duration_seconds)
        completed = subprocess.run(command)
        if completed.returncode != 0 and return_code == 0:
            return_code = completed.returncode

    return return_code


def parse_cli_args() -> tuple[list[str], int]:
    """Parse CLI arguments for manual perf test execution."""
    parser = ArgumentParser(
        description="Run grouped perf stat smoke tests for a set of events."
    )
    parser.add_argument(
        "--events",
        nargs="+",
        required=True,
        help="One or more perf event names to test manually.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Duration in seconds for the `sleep` workload. Default: 5.",
    )

    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be a positive integer")

    return args.events, args.duration


def main() -> int:
    """CLI entrypoint for manual perf test execution."""
    events, duration = parse_cli_args()
    return run_perf_test(events, duration)


if __name__ == "__main__":
    sys.exit(main())
