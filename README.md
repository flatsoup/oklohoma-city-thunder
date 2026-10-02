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
