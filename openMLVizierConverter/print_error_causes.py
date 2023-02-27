"""
Print error causes of the test runs.

Read the /result/error directory and summarize the error messages that are found in the files
"""


from pathlib import Path

import pandas as pd

DIR_ROOT = Path(__file__).resolve().parents[1]
DIR_ERROR = (DIR_ROOT / "result" / "error")


def main():
    pd.set_option('display.max_colwidth', None)
    errors = []
    for p in DIR_ERROR.iterdir():
        with open(p, 'r') as f:
            errors.append(f.read())
    errors_series = pd.Series(errors)
    max_retries = "max retries exceeded"
    errors_series = errors_series.apply(lambda v: v if max_retries not in v.lower() else max_retries)
    print(f"Total errors: {len(errors_series)}")
    if len(errors_series) > 0:
        print(errors_series.apply(lambda v: v[:150]).value_counts().to_frame(name="#"))


if __name__ == '__main__':
    main()
