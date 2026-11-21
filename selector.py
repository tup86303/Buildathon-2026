"""Select tests for each patch and assign them to workers."""

import json
from pathlib import Path

DATA = Path("/data")
WORKERS = 3


def select_tests(patch: Path, all_test_numbers: tuple[int, ...]) -> tuple[int, ...]:
    """Return the test numbers to run for this patch."""
    return all_test_numbers


def split(
    test_numbers: tuple[int, ...],
    durations: dict[int, float],
    worker_count: int,
) -> list[list[int]]:
    """Longest test first, each onto the worker with the least to do so far."""
    workers: list[list[int]] = [[] for _ in range(worker_count)]
    loads = [0.0] * worker_count
    for number in sorted(test_numbers, key=durations.__getitem__, reverse=True):
        lightest = loads.index(min(loads))
        workers[lightest].append(number)
        loads[lightest] += durations[number]
    return workers


def main() -> None:
    tests = json.loads((DATA / "tests.json").read_text())["tests"]
    durations = {test["number"]: test["seconds"] for test in tests}
    all_test_numbers = tuple(durations)
    full_schedule = split(all_test_numbers, durations, WORKERS)
    for patch in sorted((DATA / "commits").glob("*.patch")):
        selected = select_tests(patch, all_test_numbers)
        if selected == all_test_numbers:
            workers = full_schedule
        else:
            workers = split(selected, durations, WORKERS)
        print(json.dumps({"commit": patch.stem, "workers": workers}))


if __name__ == "__main__":
    main()
