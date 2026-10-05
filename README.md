## Dataset

**Source:** [fkie-cad/nvd-json-data-feeds](https://github.com/fkie-cad/nvd-json-data-feeds)
by Fraunhofer FKIE (Cyber Analysis & Defense)

**Why this source:**
The original NIST NVD legacy JSON feeds were retired in December 2023
and are now 403-blocked. This repository reconstructs the same feeds
daily from the official NVD API 2.0. The CVE records — IDs, CVSS
scores, CPE configurations — are identical to the original NIST data.

**What we downloaded:**
- File: `CVE-2021.json.xz` (decompressed and renamed to
  `nvdcve-1.1-2021.json`)
- Year: 2021 (~20,000 CVEs)
- Filter applied: CVSS ≥ 7.0 (High + Critical only)
- Anchor CVE: **CVE-2021-44228** (Log4Shell, CVSS 10.0)

**How to get the data:**
Run the included download script — it handles everything automatically:
```bash
python download_data.py
```
Or download manually from the
[releases page](https://github.com/fkie-cad/nvd-json-data-feeds/releases/latest)
→ click `CVE-2021.json.xz`

**Citation:**
> Helmke, R. (2024). *nvd-json-data-feeds: Community reconstruction
> of the legacy JSON NVD Data Feeds.* Fraunhofer FKIE.
> https://github.com/fkie-cad/nvd-json-data-feeds
>
> Underlying data: National Vulnerability Database (NVD), NIST.
> https://nvd.nist.gov
