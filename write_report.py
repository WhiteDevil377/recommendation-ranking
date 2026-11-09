"""Create aggregate figures and a report from recorded experiment results."""
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"artifacts/.mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAMES={"popularity":"Popularity","item_knn_50":"Item-KNN 50","item_knn_100":"Item-KNN 100",
       "als_32_alpha20":"ALS 32 / alpha20","als_64_alpha20":"ALS 64 / alpha20","als_64_alpha40":"ALS 64 / alpha40"}


def main():
    r=json.loads((ROOT/"artifacts/metrics.json").read_text())
    selected=r["selected_model"]
    test=r["test"][selected]
    base=r["test"]["popularity"]
    perf=r["performance"]
    ci=r["paired_ndcg_vs_popularity"]
    gain=100*(test["ndcg_at_10"]/base["ndcg_at_10"]-1)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8),gridspec_kw={"width_ratios":[1.5,1]})
    keys=list(r["test"])
    values=[r["test"][key]["ndcg_at_10"] for key in keys]
    colors=["#1768AC" if key==selected else "#8095A7" for key in keys]
    bars=axes[0].bar(np.arange(len(keys)),values,color=colors,width=.68)
    axes[0].set(xticks=np.arange(len(keys)),xticklabels=["Pop.","KNN\n50","KNN\n100","ALS32\na20","ALS64\na20","ALS64\na40"],
                ylim=(0,max(values)*1.3),ylabel="Mean test nDCG@10",title="Full-catalogue ranking on 5,034 users")
    axes[0].grid(axis="y",alpha=.16)
    for bar,value in zip(bars,values):
        axes[0].text(bar.get_x()+bar.get_width()/2,value+.0015,f"{value:.4f}",ha="center",fontsize=9)
    times=[perf["reference"]["median_ms"],perf["optimized"]["median_ms"]]
    bars=axes[1].bar([0,1],times,color=["#8095A7","#1768AC"],width=.55)
    axes[1].set(xticks=[0,1],xticklabels=["Single-user\nfull sorting","Batched\npartial top-k"],
                ylabel="Median milliseconds per 1,000 users",ylim=(0,max(times)*1.3),title="Same model and identical Top-10 lists")
    axes[1].grid(axis="y",alpha=.16)
    for bar,value in zip(bars,times):
        axes[1].text(bar.get_x()+bar.get_width()/2,value+2,f"{value:.1f} ms",ha="center",fontsize=10)
    fig.suptitle("Recommendation ranking and inference optimization",fontsize=15,fontweight="bold",y=.98)
    fig.text(.5,.015,f"Selected by validation nDCG@10. Local speedup: {perf['median_speedup']:.2f}x. Timings exclude loading and HTTP overhead.",ha="center",fontsize=9,color="#536170")
    fig.tight_layout(rect=(0,.055,1,.94),w_pad=2.8)
    (ROOT/"figures").mkdir(exist_ok=True)
    fig.savefig(ROOT/"figures/results.png",dpi=180,facecolor="white")
    plt.close(fig)
    lines=["# Measured recommendation results","","Generated on 9 November 2026 from the bundled implementation.","",
           f"Processed {r['data']['raw_ratings']:,} source ratings from {r['data']['users']:,} users. Ranked the static catalogue of {r['data']['catalogue_items']:,} movies for {r['data']['evaluated_users']:,} eligible users. {r['data']['users_excluded_from_ranking']:,} users did not meet the declared ranking criteria and are counted in the data audit.","",
           "![Ranking quality and optimization](figures/results.png)","","## Frozen model comparison","",
           "| Method | Validation nDCG@10 | Test nDCG@10 | Test Recall@10 | Test MRR@10 | Catalogue coverage |",
           "|---|---:|---:|---:|---:|---:|"]
    for key in keys:
        s=r["test"][key]
        lines.append(f"| {NAMES[key]} | {r['validation'][key]['ndcg_at_10']:.4f} | {s['ndcg_at_10']:.4f} | {s['recall_at_10']:.4f} | {s['mrr_at_10']:.4f} | {s['catalogue_coverage']:.2%} |")
    lines += ["",f"Selected **{NAMES[selected]}** by validation mean nDCG@10. The model and seen-item filter remained frozen through evaluation. ALS did not outperform the selected item-neighbour model in this experiment; its higher catalogue coverage is retained in the comparison.","",
              f"Full-catalogue test nDCG@10 increased from **{base['ndcg_at_10']:.4f}** to **{test['ndcg_at_10']:.4f}**, a **{gain:.1f}% relative increase**. Absolute paired mean difference: **{ci['mean_difference']:.4f}**, with a 95% user-bootstrap interval **[{ci['ci95_low']:.4f}, {ci['ci95_high']:.4f}]** from 1,000 resamples. This interval is conditional on the fitted models and this fixed benchmark, not an external-data guarantee.","",
              "## Optimization workload","","| Implementation | Median milliseconds | p95 across seven repetitions |","|---|---:|---:|"]
    for method in ["reference","optimized"]:
        lines.append(f"| {method} | {perf[method]['median_ms']:.2f} | {perf[method]['p95_ms']:.2f} |")
    lines += ["",f"Workload: {perf['queries']:,} fixed eligible users, k=10, batch size {perf['batch_size']}, {perf['cpu_threads']} CPU threads, alternating timing order. Median speedup: **{perf['median_speedup']:.2f}x**. Exact Top-10 list agreement: **{perf['top10_list_agreement']:.0%}**. Maximum finite score difference: **{perf['maximum_finite_score_difference']:.3g}**.","",
              f"A full batch score matrix uses {perf['score_matrix_bytes_per_full_batch']:,} bytes; this is one intermediate matrix, not total process memory. The benchmark measures a warm local ranking workload, excluding model loading, HTTP overhead, and concurrent production traffic.","",
              "## Candidate-sampling diagnostic","",
              f"On the same selected model, replacing the full catalogue with all relevant items plus 100 sampled unseen nonrelevant items per user changes nDCG@10 from **{test['ndcg_at_10']:.4f}** to **{r['sampled_negative_diagnostic']['metrics']['ndcg_at_10']:.4f}**. The sampled result is a different, easier evaluation protocol and is not the headline performance claim.","",
              "## Scope and data quality","",
              "This is a per-user temporal, transductive benchmark. Other users' training events can occur later than a particular user's evaluation events. It is not a global chronological deployment simulation. Timestamp ties are assigned to later blocks; held-out positives are disjoint from the user's training history. Lower-rated items observed before the cutoff are also filtered from recommendations.","",
              f"Training positive events: {r['data']['training_positive_events']:,}. Validation positives: {r['data']['validation_positive_events']:,}. Test positives: {r['data']['test_positive_events']:,}. Held-out test positives without a training-supported item: {r['data']['cold_item_test_events']}; these remain in the metric denominator.","",
              "Explicit ratings >=4 are transformed into positive preferences. Unobserved pairs are not proven negative preferences. No online engagement, revenue uplift, production-scale deployment, or novel algorithm is demonstrated.","",
              "## Reproducibility and sources","",f"Source archive SHA-256: `{r['data']['archive_sha256']}`.","",
              "GroupLens MovieLens 1M: https://grouplens.org/datasets/movielens/1m/. Harper and Konstan (2015), https://doi.org/10.1145/2827872. Dataset terms: https://files.grouplens.org/datasets/movielens/ml-1m-README.txt. Hu, Koren, and Volinsky (2008), Collaborative Filtering for Implicit Feedback Datasets: https://yifanhu.net/PUB/cf.pdf.","",
              "The public repository includes code and aggregate experiment outputs. Raw data and trained profile matrices are excluded; users download the source directly under GroupLens terms. No GroupLens endorsement is implied."]
    (ROOT/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print("Wrote RESULTS.md and figures/results.png")


if __name__=="__main__":
    main()
