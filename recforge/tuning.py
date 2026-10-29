"""Optional expensive Optuna search. Test data is never scored here."""
import json
from pathlib import Path
from .data import load
from .pipeline import fit
from .evaluation import evaluate


def tune(data_path, config, trials, output):
    import optuna
    data = load(data_path, config)
    def objective(trial):
        settings = dict(config, dimension=trial.suggest_categorical("dimension", [32, 48, 64]),
                        learning_rate=trial.suggest_float("learning_rate", .001, .01, log=True),
                        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True))
        engine, _, _, _ = fit(data, settings)
        frame, _, _ = evaluate(engine, data.validation, models=("two_stage",))
        warm = frame[frame.cohort == "established"]
        return float(warm["ndcg@10"].mean())
    output.mkdir(parents=True, exist_ok=False)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config["seed"]))
    study.optimize(objective, n_trials=trials)
    study.trials_dataframe().to_csv(output / "trials.csv", index=False)
    (output / "best_config.json").write_text(json.dumps(dict(config, **study.best_params), indent=2))
