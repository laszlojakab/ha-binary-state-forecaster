#!/usr/bin/env python3
"""Script to replace tuple literals with TimeKey instances in test files."""

import re


def process_file(filepath):
    """Process a file to replace tuple literals with TimeKey instances."""
    with open(filepath) as f:
        content = f.read()

    lines = content.split("\n")
    modified_lines = []

    for line in lines:
        original_line = line

        # Skip lines that already have TimeKey
        if "TimeKey(" in line:
            modified_lines.append(line)
            continue

        # Pattern 1: Standalone tuple assignment like: key = (("time_of_day", 600),)
        match = re.match(r'^(\s+)(\w+)\s*=\s*(\(\(["\'][\w_]+["\']\s*,\s*\d+\)\s*,\s*\))$', line)
        if match:
            indent, var, tuple_part = match.groups()
            line = f"{indent}{var} = TimeKey({tuple_part})"
            modified_lines.append(line)
            continue

        # Pattern 2: Multiple tuple assignments on one line
        # key1 = (("time_of_day", 600),)
        pattern = r'\b(\w+)\s*=\s*(\(\(["\'][\w_]+["\']\s*,\s*\d+\)\s*,\s*\))'
        matches = list(re.finditer(pattern, line))
        if matches:
            for match in reversed(matches):  # Process from right to left
                var, tuple_part = match.groups()
                replacement = f"{var} = TimeKey({tuple_part})"
                line = line[: match.start()] + replacement + line[match.end() :]
            modified_lines.append(line)
            continue

        # Pattern 3: Function call arguments: model.distribution((("...", ...),))
        # or assert statements with tuples
        pattern = r'\b(distribution|update_duration|assert\s+[^=]*\s+)(\(\(["\'][\w_]+["\']\s*,\s*\d+\)\s*,\s*\))'
        if re.search(pattern, line):
            line = re.sub(r'(\(\(["\'][\w_]+["\']\s*,\s*\d+\)\s*,\s*\))', r"TimeKey(\1)", line)
            modified_lines.append(line)
            continue

        # Pattern 4: in operator: (("...", ...),) in model._states or model._stats.stats
        if " in model._states" in line or " in model._stats.stats" in line:
            line = re.sub(
                r'(\(\(["\'][\w_]+["\']\s*,\s*\d+\)\s*,\s*\))(\s+in\s+)', r"TimeKey(\1)\2", line
            )
            modified_lines.append(line)
            continue

        # Default: keep original line
        modified_lines.append(line)

    result = "\n".join(modified_lines)
    return result


if __name__ == "__main__":
    filepath = "tests/test_discrete_conditional_model.py"
    result = process_file(filepath)

    # Write to new file
    with open("tests/test_discrete_conditional_model_new.py", "w") as f:
        f.write(result)

    print("Done! File saved to tests/test_discrete_conditional_model_new.py")
    print("Review the changes, then:")
    print(
        "  mv tests/test_discrete_conditional_model_new.py tests/test_discrete_conditional_model.py"
    )
