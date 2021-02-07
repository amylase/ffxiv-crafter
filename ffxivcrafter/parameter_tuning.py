import optuna

from ffxivcrafter.crafter_trainer import train_model, eval_model


def objective(trial: optuna.Trial) -> float:
    hyper_params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1.0, log=True),
        "epsilon": trial.suggest_float("epsilon", 0, 1),
        "gamma": 0.999
    }
    model = train_model(**hyper_params)
    return eval_model(model)


def hyperparameter_tuning():
    study = optuna.create_study(
        sampler=optuna.samplers.RandomSampler(),
        pruner=optuna.pruners.MedianPruner(),
        direction="maximize"
    )
    study.optimize(objective, n_trials=50)
    print(study.best_trial.params)
    best_model = train_model(**study.best_params, train_iter=50000)
    print(eval_model(best_model))