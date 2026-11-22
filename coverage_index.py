"""Invert baseline test coverage once so queries start from the changed lines."""

import json
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path


def invert(source):
    tests = sorted(source)
    by_line = {}
    for i, test in enumerate(tests):
        bit = 1 << i
        for line in source[test]:
            by_line[line] = by_line.get(line, 0) | bit
    lines = sorted(by_line)
    masks = list(dict.fromkeys(by_line[line] for line in lines))
    position = {mask: i for i, mask in enumerate(masks)}
    return [tests, lines, [position[by_line[line]] for line in lines],
            [format(mask, "x") for mask in masks]]


class LineIndex:
    def __init__(self, data):
        self.tests, self.lines, self.offsets, masks = data
        self.masks = [int(mask, 16) for mask in masks]

    def __iter__(self):
        return iter(self.tests)

    def covering(self, regions):
        selected = 0
        for start, end in regions:
            lo = bisect_left(self.lines, start)
            hi = bisect_right(self.lines, end, lo)
            for i in range(lo, hi):
                selected |= self.masks[self.offsets[i]]
        result = set()
        while selected:
            bit = selected & -selected
            result.add(self.tests[bit.bit_length() - 1])
            selected ^= bit
        return result


if __name__ == "__main__":
    data = json.loads(Path(sys.argv[1]).read_bytes())
    data["line_index"] = {p: invert(v) for p, v in data.pop("source").items()}
    Path(sys.argv[2]).write_bytes(json.dumps(data, separators=(",", ":")).encode())
