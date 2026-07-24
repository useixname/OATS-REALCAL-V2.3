# Data preparation

REAL-CAL-V2.3 is a real-data-calibrated semi-synthetic benchmark. The release
does not redistribute the underlying third-party files. Obtain the datasets
from their original providers under their respective terms, then place the
extracted files in this layout:

```text
data_real/
  raw/
    tdrive/
      extracted/
        *.txt
    beijing_air/
      extracted/
        PRSA_Data_20130301-20170228/
          *.csv
    purpleair_epa/
      extracted/
        Data_DevelopmentUSPAcorrection_210408/
          Full24hrdataset.csv
```

Build the frozen calibration-profile format:

```bash
python scripts/build_realcal_profile.py
```

The command writes:

```text
data_real/REAL-CAL-V1/calibration_profile.json
```

REAL-CAL-V2 uses this real-data calibration profile and the version-2 value
scale in the trace generator. Generate the ten paired manuscript seeds with:

```bash
python scripts/generate_realcal_trace.py \
  --dataset-version 2 \
  --seeds 20260715 20260716 20260717 20260718 20260719 \
          20260720 20260721 20260722 20260723 20260724 \
  --workers 0
```

The generated traces and hash manifest are written below `data/REAL-CAL-V2/`.
Generation refuses to overwrite existing seed directories.
