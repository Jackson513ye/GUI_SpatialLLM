# Running the Pipeline

## Requirements

* **Python:** 3.12
* **Shell:** `bash` (Linux, macOS, WSL, or Git Bash on Windows)
* **PolyFit:** Download a wheel for Python 3.12 from [LiangliangNan/PolyFit Releases](https://github.com/LiangliangNan/PolyFit/releases)
* **Input files:**

  ```
  data/input/deSkatting.e57
  data/input/deSkatting_segmented.las
  ```

---

## 1️ Set up environment

### Linux / macOS / WSL

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install /path/to/PolyFit-<version>-cp312-*.whl
```

### Windows (Git Bash)

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install C:/path/to/PolyFit-<version>-cp312-*.whl
```

---

## 2️ Place input data

```
data/input/deSkatting.e57
data/input/deSkatting_segmented.las
```

---

## 3️ Run the workflow

From the repository root (virtual environment active):

```bash
snakemake --cores all --configfile config.yaml
```

If `snakemake` is not found:

```bash
python -m snakemake --cores all --configfile config.yaml
```

---

## 4️ Output

Results are written under:

```
data/output/
```

including generated PLYs, BVG/OBJ/CSV files, and the final marker:

```
data/output/_PCG_DONE
```

---

## Notes

* **Linux / macOS / WSL:** recommended environment.
* **Windows:** run **inside Git Bash** or **WSL** (PowerShell/CMD will fail because `bash` is required).
* The pipeline automatically handles panoramas, class splits, room segmentation, PolyFit, and PCG processing based on `config.yaml`.

---

 **Checklist**

* [ ] Python 3.12 venv active
* [ ] Dependencies + PolyFit wheel installed
* [ ] Input `.e57` and `.las` files in `data/input/`
* [ ] Run: `snakemake --cores all --configfile config.yaml`

---
