import wandb

api = wandb.Api()

sweep = api.sweep("moorekevin-/vision-transformers-cifar10/sweeps/jo541euz")

runs = sweep.runs
top_runs = sorted(runs, key=lambda r: r.summary.get(
    'val_acc', 0), reverse=True)[:3]

for i, run in enumerate(top_runs, 1):
    print(f"\nTop {i}: {run.name}")
    print(f"val_acc: {run.summary['val_acc']:.4f}")
    print("Hyperparameters:")
    for k, v in run.config.items():
        print(f"  {k}: {v}")
