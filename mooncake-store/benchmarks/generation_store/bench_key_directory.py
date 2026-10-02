"""Include an arbitrary string-key directory in the compact-index cost model.

Synthetic Python comparison, not production C++ metadata or trainer tracing.
Key fixture mirrors PoolKey's model/rank/group/hash fields. Keys and queries
are generated outside timing; query strings are fresh, including hash cost.
"""

import argparse
import array
import hashlib
import json
from pathlib import Path
import random
import sys
from bench_generation import timed


def deep_size(value, seen):
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(deep_size(k, seen) + deep_size(v, seen) for k, v in value.items())
    elif isinstance(value, tuple):
        size += sum(deep_size(v, seen) for v in value)
    return size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    rows = []
    for count in [1024, 65536]:
        keys = [
            "PolicyModel@tp:0@pcp:0@dcp:0@pp:0@group:0@"
            + hashlib.sha256(str(i).encode()).hexdigest()
            for i in range(count)
        ]
        rng = random.Random(7743)
        shapes = {
            "dense_prefix": list(range(min(4096, count))),
            "uniform_sparse": rng.sample(range(count), min(4096, count)),
        }
        for pattern, ids in shapes.items():
            expected = sum(i * 458752 + 458752 for i in ids)
            for repeat in range(5):
                for compact in [False, True] if repeat % 2 == 0 else [True, False]:
                    if compact:

                        def build():
                            directory = {key: i for i, key in enumerate(keys)}
                            index = array.array(
                                "Q",
                                (v for i in range(count) for v in (i * 458752, 458752)),
                            )
                            return directory, index

                    else:

                        def build():
                            return (
                                {
                                    key: (i * 458752, 458752)
                                    for i, key in enumerate(keys)
                                },
                            )

                    state, creation = timed(build)
                    footprint = deep_size(state, set())
                    query = [keys[i].encode().decode() for i in ids]
                    if compact:
                        directory, index = state

                        def lookup():
                            total = 0
                            for key in query:
                                ordinal = directory[key]
                                total += index[ordinal * 2] + index[ordinal * 2 + 1]
                            return total

                    else:
                        mapping = state[0]

                        def lookup():
                            total = 0
                            for key in query:
                                offset, length = mapping[key]
                                total += offset + length
                            return total

                    total, measured = timed(lookup)
                    assert total == expected
                    rows.append(
                        dict(
                            count=count,
                            pattern=pattern,
                            queries=len(query),
                            repeat=repeat,
                            compact=compact,
                            footprint_bytes=footprint,
                            build_cpu_ns=creation["cpu_ns"],
                            lookup_cpu_ns=measured["cpu_ns"],
                            lookup_wall_ns=measured["wall_ns"],
                        )
                    )
    (args.output / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("Completed", len(rows), "string-key directory samples")


if __name__ == "__main__":
    main()
