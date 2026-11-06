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

**The exact file is not included here, for privacy.** Some descriptions contain names, addresses, e-mail addresses or phone numbers written by citizens, so the texts are not republished in this repository. The author keeps the exact file (`CivicLens_Open311.csv`, the SHA-256 above) and can share it on request.

The portal is updated daily, so a new download will usually be **newer** and have a different checksum and slightly different counts. To reproduce the exact numbers in the paper, use the original file and check it with:

```
sha256sum CivicLens_Open311.csv      # Linux / Colab
shasum -a 256 CivicLens_Open311.csv  # macOS
```
