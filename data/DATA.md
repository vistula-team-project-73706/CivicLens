# Dataset

**Name:** Open311 (uReport service requests), City of Bloomington, Indiana, USA

**Dataset ID:** `aw6y-t4ix`

**Portal page:** https://data.bloomington.in.gov/d/aw6y-t4ix

**Direct CSV download:** https://data.bloomington.in.gov/api/views/aw6y-t4ix/rows.csv?accessType=DOWNLOAD

**Licence:** Open Data Commons Public Domain Dedication and License (PDDL 1.0). Free reuse is allowed.

**Fields used:**
- `description` (text) is the model input.
- `service_name` (category) is the label.
- No other fields are used.

## The exact file used in the paper

| Property | Value |
|---|---|
| Downloaded | 5 November 2026 (inside Google Colab) |
| Size | 48,969,280 bytes |
| Rows | 130,228 (67 categories, 0 missing labels, 9,301 empty descriptions) |
| SHA-256 | `802e12249d4e088f3efc813378dd9e1fe126733945ccc2c1b1f814eb1354cc97` |

## Data privacy and access

The original dataset was obtained from the City of Bloomington Open Data portal (link above). A local snapshot was retained for reproducibility and is identified by its SHA-256 checksum (table above).

The original dataset is not redistributed in this repository because it contains personal information, including report descriptions and location-related information.

Researchers can obtain the source dataset directly from the original provider, verify the downloaded snapshot using the published checksum, and reproduce the analysis using the supplied scripts and instructions. Note that the portal is updated daily, so a new download will usually be newer than the snapshot used in the paper, with a different checksum and slightly different counts.

Released prediction files contain anonymous group identifiers rather than original citizen report text.

Check a downloaded file with:

```
sha256sum CivicLens_Open311.csv      # Linux / Colab
shasum -a 256 CivicLens_Open311.csv  # macOS
```
