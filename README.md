# OpenML Vizier Converter
The OpenML Vizier Converter is a Python library for connecting OpenML with Google Vizier.

## Prerequisites
Python 3.10+

## Install

```commandline
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```
## Usage
To test the conversion on all run ids:
```commandline
python3 openMLVizierConverter
```

To use the convertor in your code:
```python3
import pandas as pd
from pathlib import Path

import converter

DIR_ROOT = Path(__file__).resolve().parents[1]
df_ids = pd.read_csv(DIR_ROOT / "data" / "runids.csv")
ids = set(df_ids['id'].unique())

for id_ in ids:
    try:
        study = converter.create_study_from_openml(id_)
    except:
        continue # 44 / 22359 runs will throw an error. These can be ignored.
    # do something with the study
```
