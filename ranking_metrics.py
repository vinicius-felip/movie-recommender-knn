from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


OUT = Path(__file__).resolve().parent / "results_v2"


def precision_recall_at_n(predictions, n=5, threshold=4):

    precisions = []
    recalls = []

    for _, group in predictions.groupby("user_id"):

        relevant = (group["actual"] >= threshold).sum()

        if relevant == 0:
            continue

        recommended = group.sort_values(
            ["predicted", "movie_id"],
            ascending=[False, True]
        ).head(n)

        hits = (recommended["actual"] >= threshold).sum()

        precisions.append(hits / n)
        recalls.append(hits / relevant)

    return {
        f"precision@{n}": (
            sum(precisions) / len(precisions)
            if precisions else float("nan")
        ),
        f"recall@{n}": (
            sum(recalls) / len(recalls)
            if recalls else float("nan")
        ),
        "users_evaluated": len(recalls)
    }


def generate_chart(results, columns, title, filename):

    data = results.set_index("model")[columns] * 100

    ax = data.plot(
        kind="bar",
        figsize=(10, 6),
        width=0.7
    )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Modelo k-NN")
    ax.set_ylabel("Percentual (%)")
    ax.set_ylim(0, 100)

    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax.legend(
        title="Métricas",
        loc="upper right"
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f%%",
            padding=3,
            fontsize=9
        )

    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        OUT / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def main():

    predictions = pd.read_csv(
        OUT / "predictions.csv"
    )

    rows = []

    for model, group in predictions.groupby("model"):

        row = {
            "model": model,
            "metric": group["metric"].iloc[0],
            "k": group["k"].iloc[0]
        }

        for n in [5, 10]:

            metrics = precision_recall_at_n(
                group,
                n=n,
                threshold=4
            )

            row.update(metrics)

        rows.append(row)

    results = pd.DataFrame(rows)

    results.to_csv(
        OUT / "ranking_metrics.csv",
        index=False
    )

    print("\n=== RESULTADOS DO RANKING ===\n")

    print(results.to_string(index=False))

    # Gráfico de Precision
    generate_chart(
        results,
        ["precision@5", "precision@10"],
        "Comparação de Precision por modelo",
        "precision.png"
    )

    # Gráfico de Recall
    generate_chart(
        results,
        ["recall@5", "recall@10"],
        "Comparação de Recall por modelo",
        "recall.png"
    )

    # Gráfico com todas as métricas
    generate_chart(
        results,
        [
            "precision@5",
            "recall@5",
            "precision@10",
            "recall@10"
        ],
        "Desempenho dos sistemas de recomendação",
        "precision_recall.png"
    )

    print(f"\nResultados salvos em: {OUT}")
    print("Gráficos gerados com sucesso!")


if __name__ == "__main__":
    main()