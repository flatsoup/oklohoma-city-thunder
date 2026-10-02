# AI Prompts Log

This project was completed with the assistance of Claude (Anthropic), run through the
Claude Code CLI. Per the project brief, below is the log of prompts used to produce
`project_code.py`, `project_writeup.pdf`, and `submission.csv`.

All code, feature-engineering decisions, modeling choices, and the writeup were reviewed
and understood by me before submission; the AI was used as a coding/analysis assistant,
not as an unreviewed black box.

---

## Prompt 1 (initial task / plan)

> мне нужно чтобы выполнили это задание. для этого прочитай задание, создай план выполнения
> а затем в конце собери чисто для меня туториал простыми словами что и как сделано. чтобы
> можно понимать что происходит, а также чтобы можно было самому сесть и научиться своими
> руками. логируй мне ключевые действия.
>
> Наш план действий (выполняем пошагово):
>
> 1. Feature Engineering (Инжиниринг признаков):
>    - Рассчитай точное расстояние и угол до центра кольца (x = -41.75, y = 0) по имеющимся
>      координатам locationx и locationy.
>    - Извлеки динамические фичи из массива приближения защитника (closestdefapproach:
>      расстояния за 1.0s, 0.75s, 0.5s, 0.25s) — скорость закрытия (closing speed),
>      ускорение защитника.
>    - Рассчитай процент попаданий (shooting efficiency / FG%) для игроков (shooter_id),
>      типов бросков (shottype) и зон кортов, используя Out-of-Fold / Target Encoding с
>      K-Fold кросс-валидацией, чтобы избежать утечки данных (data leakage).
>    - Сформируй взаимодействия признаков: например, distance * closestdefdist,
>      dribblesbefore * shotclock, количество и дистанции опекунов (distcont1-distcont4,
>      num_contesters).
> 2. Моделирование и Валидация:
>    - Предложи архитектуру ансамблирования (LightGBM / XGBoost / CatBoost или Logistic
>      Regression со сплайнами) для минимизации Log-Loss.
>    - Настрой Stratified Group K-Fold валидацию (с группировкой по сезонам season_id или
>      датам/матчам, если нужно).
> 3. Генерация результатов:
>    - Напиши чистый Python-код для формирования файла submission.csv.
>    - Построй график важности признаков (Feature Importance / SHAP values).
> 4. Структура Writeup (Отчёта):
>    - Составь план и текстовые черновики для project_writeup.pdf, чтобы объяснить
>      бизнес-логику решения и показать глубокое понимание баскетбола и DS-процессов.
>
> Начни с первого шага: Напиши базовый Python-скрипт (Data Pipeline) на polars/pandas +
> scikit-learn/lightgbm, который загружает training.csv.gz, выполняет первичную очистку,
> генерирует ключевые баскетбольные признаки и подготавливает кросс-валидацию по Log-Loss.
>
> если что-то в промпте противоречит тому что ты прочитал в файлах, ты можешь изменять
> смело, но с объянением почему

**What the assistant did with this prompt:** read `Analyst Intern Project Brief.pdf`,
inspected `training.csv.gz` / `testing.csv.gz` directly (column overlap checks, nulls,
value ranges), then built `project_code.py` implementing the full pipeline described
above, with one deliberate deviation from the plan (see below) explained to the user.

**Deviation made and why:** the plan asked for "Stratified Group K-Fold validation,
grouped by `season_id`." Direct inspection showed `training.csv` contains only two
season IDs (`fe055`, `2676a`) and `testing.csv` contains a third season ID (`d7a2d`) that
never appears in training at all. Consequences:
  - `season_id` cannot be used as a raw predictive feature (always an unseen category at
    test time).
  - A 2-group "Group K-Fold by season" isn't a meaningful cross-validation scheme.

  Replacement: standard `StratifiedKFold` (on `outcome`) for model training and for the
  out-of-fold target encoding, plus a separate train-on-one-season / validate-on-the-other
  diagnostic (`season_holdout_diagnostic` in `project_code.py`) that estimates how much
  log-loss degrades under a season shift — which is exactly the situation the real
  `testing.csv` evaluation represents.

---

## Summary of autonomous build steps taken from Prompt 1

Working from Prompt 1 and the one deviation noted above, the assistant:
1. Inspected training.csv.gz / testing.csv.gz directly (column overlap, null counts,
   value ranges, parsed the `closestdefapproach` string format) before writing any code.
2. Implemented `project_code.py`: data loading/cleaning, geometry features, defender
   closeout dynamics, contester aggregates, leakage-safe out-of-fold target encoding,
   interaction terms, a 5-fold Stratified K-Fold LightGBM model, a spline logistic
   regression baseline, an OOF-log-loss-optimized blend search, a season-holdout
   diagnostic, feature importance / SHAP-style / calibration / zone-FG% plots, and the
   final `submission.csv` writer.
3. Ran the full pipeline end-to-end, validated `submission.csv` (row order, column names,
   no nulls/out-of-range values preserved against the original template) and inspected
   all generated figures before using them in the writeup.
4. Wrote `project_writeup.pdf` (8 pages: objective, data findings, feature engineering,
   modeling/ensembling, validation strategy, results/diagnostics, reflections/future
   work) using the real metrics and figures produced by `project_code.py`, generated via
   a one-off `fpdf2` script (not part of the submission) rather than a required library.
   Rendered the PDF to images (`pdftoppm`) to visually QA layout before finalizing; found
   and fixed a cursor-position bug in that generator script (a `multi_cell(width=0, ...)`
   call was leaving the draw cursor at the right margin instead of returning it to the
   left margin, corrupting the next table row) before producing the final PDF.

Final headline numbers (see `project_writeup.pdf` Section 6 for full detail): naive
baseline log-loss 0.6896, logistic regression + splines 0.6342, LightGBM (5-fold OOF)
0.6243 (final submission model), season-holdout diagnostic 0.6298 / 0.6253.

**Outstanding item for the user to complete manually:** the top-of-file comment in
`project_code.py` currently reads `NUMBER: TODO_FILL_LAST4DIGITS` -- the assistant does
not have access to the phone number used on the application and left this as an explicit
placeholder rather than guessing.
