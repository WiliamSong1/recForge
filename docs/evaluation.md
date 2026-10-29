# Evaluation contract

The first benchmark freezes `configs/default.json` before inspecting test metrics.
Further tuning belongs to a new experiment, with a new untouched temporal window or an explicitly acknowledged reused test set.

## Splits and population

MovieLens 1M is sorted by timestamp. The earliest approximately 80% of interactions form training; the next 10% validation; the last 10% test. Every timestamp belongs to one partition. All interactions at the cutoff remain in the earlier partition. Exact cutoffs, row counts, and a SHA-256 of the ratings file are saved in each report. Duplicate user–item observations retain their earliest value, so later revisions cannot overwrite historical training labels.

The evaluation universe consists of all movies observed in training. We exclude each user's training interactions, regardless of rating. Relevance is binary: a held-out rating of at least 4. We exclude unavailable items from the relevance denominator and report how many positives were excluded. Users with no remaining positive are excluded from ranking metrics and counted. Precision divides by 10 even for a short list. Recall divides by the user's eligible positive count. NDCG uses the ideal ordering of up to ten relevant items. All metrics are macro averages across users; coverage is the fraction of the eligible catalog recommended at least once.

Users with fewer than five training interactions form a separate cold cohort and receive popularity fallback for every model. The established cohort is the primary personalized-model comparison. Static test-time history is the original training snapshot: validation interactions do not update test features or exclusions. This measures a frozen service over a later time window, not online learning. Global temporal splits can produce substantial user and popularity shift; per-user leave-one-out benchmarks are not comparable.

User IDs from all partitions are mapped to tensor indices, but this only preallocates embeddings. No held-out rating or activity statistic enters a training feature. Unknown-user embeddings are never used: cold users fall back. MovieLens 1M has no tags; no tag features are invented. Movie metadata is treated as static catalog information; serving candidates must still have a training observation.

## Objectives and selection

Popularity counts positive training interactions. Matrix factorization minimizes squared rating error with AdamW regularization, user/item biases, and a global mean. Its predictions are clipped to [1, 5] only for RMSE/MAE and application rating outputs; ranking uses raw scores. Its training objective is different from the two-tower objective, so this is a comparison of systems, not an isolated architecture ablation.

The two-tower model combines item embeddings and multi-hot genres through an MLP; the user tower transforms user embeddings through another MLP. Both outputs are L2-normalized. BPR uses positives rated at least 4 and uniformly sampled items from the training-observed catalog. All known training interactions, including dislikes, are excluded from negatives. Future positives are unknown at training time and are not consulted; some implicit negatives can therefore be future positives. Fixed validation negatives exclude validation positives and are used only for early stopping.

MF early stopping uses temporal validation rating MSE; two-tower stopping uses temporal validation BPR loss. The first implementation does not tune model checkpoints directly on NDCG. The Optuna entry point searches only validation scores and never calls test evaluation. Model capacity and optimizer budgets are deliberately modest, not claims of best-achievable performance.

## Temporal reranking

Split the training period chronologically again, 80/20. Train temporary MF and two-tower models on the earlier portion. Generate ANN candidates and features exclusively from that earlier history. Train a histogram gradient-boosted classifier on the later training portion, using ANN hard negatives and the union of candidate items and eligible positive labels. Adding missed positives during ranker training prevents the ranker from learning only easy retrieval successes; evaluation never injects positives into candidates.

Features are two-tower affinity, MF rating, genre affinity, log popularity, smoothed item rating, log history length, and smoothed user rating. Classifier outputs are ranking scores, not calibrated preference probabilities or predicted ratings. The final base models are then refit on the full training period, giving every evaluated model the same available training history. Applying the temporal ranker to refit features introduces distribution shift, explicitly tracked in metadata. A future improvement is cross-fitted ranker training across several temporal folds.

Outer validation selects a blend of retrieval and reranker percentile ranks from {0, .25, .5, .75, 1}; ties favor less reranking. Test results include the chosen blend and pure reranking separately. A zero selected blend means no evidence supporting reranking in that experiment. Report that outcome rather than claiming a two-stage lift.

## Retrieval and uncertainty

Exact two-tower scores every eligible unseen item. ANN uses FAISS HNSW, inner product over normalized embeddings, M=32, efConstruction=120, efSearch=128. HNSW builds and searches use one FAISS thread for deterministic construction and to avoid an observed macOS OpenMP deadlock. Candidate pools default to 100. Oversampling accounts for seen items, and all returned IDs are filtered and deduplicated. ANN recall compares unseen candidates with FAISS exact search at pools 20, 100, and 300. At this small catalog size ANN overhead can exceed exact-search cost.

Paired bootstrap intervals resample the same held-out users for each pair of systems. They describe user-sampling uncertainty conditional on one trained model; they do not estimate training-seed uncertainty. The aggregation script reports mean and sample standard deviation across explicitly separate training seeds. There is no claim of statistical significance from one point estimate.

Offline timings include filtering and scoring on the local process. The serving benchmark uses real loopback HTTP and includes profile lookup, cache behavior, and event persistence. It is a distinct workload using adapted application profiles. Do not treat its recommendation quality as measured by the training-user benchmark. Peak RSS covers the entire training process, not isolated inference memory. Hardware contention and OS cache state can influence measurements.
