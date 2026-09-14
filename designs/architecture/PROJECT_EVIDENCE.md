# Current portfolio projects — 2026-09-14

## Selection and role

- User chose to remove Rust/tinydb from the visible portfolio.
- `standard-medallion-apache-ecosystem-on-nix` replaces the earlier CDC/Iceberg card; the old diagram source is retained but excluded from the public bundle/tabs.
- Added `datn_outlier_hs_nlmt`; the owner explicitly supplied the role **Data & AI Lead**. This is credited as a team project, **The Outliers**, not sole authorship.

## Medallion on Nix

GitHub main and clean local checkout both at `9b671f3dd6b4b7e2d73e2d37142ce4e6e720eec5` during inspection.
Source: `/home/tandat/Desktop/cv/DE/standard-medallion-apache-ecosystem-on-nix`.

- `scripts/flink_bronze_ingest.py`: five Kafka topic sources and NDJSON FileSink. Current active processor hardcodes offset/partition metadata and drops malformed JSON, so no claims of full lineage or working DLQ are published.
- `scripts/spark_bronze_to_silver_*.py`: domain transforms, window deduplication, Parquet output.
- `scripts/spark_silver_to_gold_*.py`: KPI aggregation code and Parquet output. IoT anomaly code explicitly includes a placeholder path.
- `scripts/spark_gold_to_postgres_*.py`: JDBC append loaders exist, but README lines 51–52 leave the load run and Airflow orchestration pending.
- `.github/workflows/test-postgres-load.yml` applies schema and lists tables; it does not establish successful end-to-end data loading.
- `ETL/milestones/001-flink-bronze-ingest-success.md`: recorded job submission; pending items do not prove completed checkpoints or output correctness.
- Original `docs_images/medallion_architecture.svg` copied unchanged to `public/architecture/medallion-original.svg` and linked beside the new focused diagram.
- No demo video found in inspected README/tutorial/milestone files. Do not borrow the Oracle video.

## Solar PV graduation project

GitHub main and clean local checkout both at `0b49f9eb637b7c85305a998a8e2c876870d367f4` during inspection.
Source: `/home/tandat/Desktop/cv/DS/datn_outlier_hs_nlmt`.

- `srcs/06_run_pipeline/main.py`: module orchestration for loading staging, buffer transforms, imputation, anomaly flags, warehouse and data marts.
- `srcs/00_database/sql/create_datawarehouse.sql`: two fact tables and five dimensions.
- `srcs/05_machine_learning/forcasting_pipeline/README.md`: separate CLI forecasting stages, LightGBM, evaluation and SHAP; no metrics copied into the project card.
- `srcs/07_dashboard/streamlit_app/app.py`: two actual page entries, ML and What-if. Old README paths/pages differ, so the portfolio does not repeat those startup instructions.
- Copied unchanged screenshots: `reports/figures/Dashboard_overview.png` → `solar-dashboard.png`; `reports/images/streamlit_1_timeseries.png` → `solar-forecast.png`.
- Screenshots are labeled as saved report snapshots. Numbers belong to the scope shown inside the image, not a new benchmark claim.
- The older `reports/figures/system_architecture.png` has horizon/feature-count claims that differ from other source versions; it was inspected but not republished as the current architecture. The new SVG keeps a high-level code-backed flow with no such numeric claims.
- No demo video found in the inspected entry documentation. The card accurately says dashboard/results images rather than video.

Original source repositories were inspected read-only: no data pipeline, database, training, migration, Git commit, or deployment was run.

## Solar presentation refinement

- Explicit GMM–IF, LightGBM, Optuna and SHAP nodes. `actions/tune_optuna.py` is a standalone CV/TPE action; the forecasting README section 4 distinguishes trial output `best_params_optuna_thu.json` from manually approved `best_params.json`.
- Added unchanged `reports/images/streamlit_2_shap.png` as `solar-shap.png`, showing global and local explanation results.
- Final requested layout: separate sibling tabs `Solar PV / Architecture` and `Solar PV / Dashboard`. No large dashboard preview in the project card, and no gallery inside the diagram panel. The card's results link opens the dedicated dashboard tab.
- Neovim playground is a browser-only subset of the owner's local keymaps: Space leader, Space ff, Space n, Ctrl+s and ;;. Catppuccin Mocha and pink cursor are derived from the inspected options/colorscheme. No private config files, paths, plugins or shell execution are published by the playground.
