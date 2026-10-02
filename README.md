# Analyst Intern Project

Welcome to the next phase of the hiring process: a technical project. Please read `Analyst Intern Project Brief.pdf` for instructions.

**Make sure you press "Submit Project"** in the email once you are finished and have committed all requisite files to the main branch.

We recommend cloning this repository to your machine. See the guide [here](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) if you need help.

**Do not unzip** the training/testing `.csv.gz` files. You can read them in directly with the following:
- R:
  ```
  library(readr)
  train_df <- read_csv("training.csv.gz")
  ```
- Python:
  ```
  import polars as pl
  train_df = pl.read_csv("training.csv.gz")
  ```

---

## Repository layout

The four submission files live at the top level, as the brief requires:

| File | What it is |
|---|---|
| `project_code.py` | Full pipeline: features, models, CV, plots; fills `submission.csv` |
| `submission.csv` | Predictions (`make_prob`) for every test shot |
| `project_writeup.pdf` | Project writeup |
| `ai_prompts.md` | Log of AI prompts used |

Everything else is grouped by role:

```
scripts/            helper code for the writeup
  writeup_content.py    writeup text (single source for PDF and Word)
  build_writeup.py      renders project_writeup.pdf + outputs/project_writeup.docx
outputs/            everything the code produces
  figures/              charts used in the writeup
  metrics_summary.json  final scores from project_code.py
  project_writeup.docx  Word version of the writeup
notebooks/
  eda.ipynb             early data exploration
```

To reproduce:

```bash
pip install -r requirements.txt
python project_code.py              # trains, writes submission.csv + outputs/
python scripts/build_writeup.py     # rebuilds the PDF and Word writeups
```
