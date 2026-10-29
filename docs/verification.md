# Verification and milestone status

Verified locally on Python 3.12.15, macOS ARM64:

- Phase 1: official MovieLens 1M ingestion; chronological baseline evaluation; popularity and rating MF trained on three seeds; complete raw ranking and rating metrics saved.
- Phase 2: PyTorch two-tower training, negative sampling, temporal early stopping, saved checkpoints, and comparisons completed on three seeds. A real MLflow run finished with logged parameters, metrics, and artifacts. Optuna search is implemented; a full search study remains unrun.
- Phase 3: FAISS HNSW retrieval, filtering, temporal learned reranker, validation blend selection, test evaluation, exact/approximate fidelity and speed comparisons, and paired confidence intervals completed. The selected blend helps; pure reranking does not.
- Serving slice: persistent local profiles, bearer-token ownership, rating updates, recommendations, feedback events, model metadata, readiness, caching, and Prometheus instrumentation tested. Real HTTP workloads exercised three models at concurrency 1/4/8 with cold/warm application caches.
- `python -m pytest -q`: **14 passed**. Tests include a full synthetic pipeline run and future-label invariance. Dependency warnings concern Starlette's httpx integration and unavailable hardware introspection, not failed assertions.
- Python compilation checks passed for the package, tests, and benchmark scripts. All three saved inference versions load successfully. Benchmark charts were visually inspected.

The first attempted multithreaded HNSW build stalled. Stack traces identified FAISS index insertion; single-threaded construction resolved it, and all reported full runs use the fix. One earlier noninteractive diagnostic process could not be terminated from the sandbox. At handoff its PID was 51211; it owns `work/training-42.log`. The later diagnostic process and the HTTP server were stopped. Verify the PID before stopping it from your own terminal if it remains present.

Not validated or completed: deployment Docker build, GitHub-hosted CI execution, PostgreSQL/Redis, frontend, full Optuna study, model promotion/rollback, scheduled retraining, drift monitoring, online quality, EC2, or MovieLens 32M. CPU brand and installed RAM were unavailable through the sandbox. The README and benchmark report distinguish implemented local functionality from these remaining production milestones.

`RecForge-source.zip` contains source, tests, configs, documentation, and measured report/plot files. It excludes raw MovieLens data, environments, training binaries, local databases, and MLflow's local store. The adjacent local `RecForge/` directory retains downloaded data and trained artifacts for immediate use.
