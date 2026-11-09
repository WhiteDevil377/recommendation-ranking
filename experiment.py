"""Compare fixed recommendation models, then benchmark ranking optimization."""
import csv
import importlib.metadata as metadata
import json
from time import perf_counter
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from prepare import ROOT,prepare
from recommenders import Ranker,fit_als,item_similarity,ranking_metrics


def write_csv(path,rows):
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(model,users,relevant):
    rankings=model.recommend_batch(users)
    per_user=[ranking_metrics(ranking,relevant[int(user)]) for user,ranking in zip(users,rankings)]
    mean={key:float(np.mean([row[key] for row in per_user])) for key in per_user[0]}
    mean["catalogue_coverage"]=float(len(np.unique(rankings))/model.B.shape[1])
    return mean,per_user


def main():
    data=prepare()
    artifact=ROOT/"artifacts"
    artifact.mkdir(exist_ok=True)
    (artifact/"checkpoints").mkdir(exist_ok=True)
    (artifact/"data_quality.json").write_text(json.dumps(data["info"],indent=2))
    B,seen,users=data["train"],data["seen"],data["eligible"]
    popularity=np.asarray(B.sum(axis=0)).ravel()
    models={}
    validation={}
    fit_times={}
    objectives={}
    with threadpool_limits(limits=2):
        candidates=[("popularity",None), ("item_knn_50",50),("item_knn_100",100),
                    ("als_32_alpha20",(32,20)),("als_64_alpha20",(64,20)),("als_64_alpha40",(64,40))]
        for name,config in candidates:
            start=perf_counter()
            if name=="popularity":
                model=Ranker(name,B,seen,popularity)
            elif name.startswith("item_knn"):
                similarity=item_similarity(B,config)
                model=Ranker(name,B,seen,popularity,similarity=similarity)
            else:
                X,Y,history=fit_als(B,factors=config[0],alpha=config[1])
                model=Ranker(name,B,seen,popularity,X=X,Y=Y)
                objectives[name]=history
            fit_times[name]=perf_counter()-start
            models[name]=model
            validation[name],_=evaluate(model,users,data["validation"])
            joblib.dump(model,artifact/"checkpoints"/f"{name}.joblib",compress=3)
            print("VALIDATION",name,json.dumps(validation[name]),flush=True)
        selected=max(validation,key=lambda name:validation[name]["ndcg_at_10"])
        (artifact/"frozen_selection.json").write_text(json.dumps({"selected_model":selected,"criterion":"validation mean nDCG@10"},indent=2))
        print("FROZEN_SELECTION",selected,flush=True)
        test={}
        individual={}
        for name,model in models.items():
            test[name],individual[name]=evaluate(model,users,data["test"])
            print("TEST",name,json.dumps(test[name]),flush=True)
        difference=np.array([a["ndcg_at_10"]-b["ndcg_at_10"]
                             for a,b in zip(individual[selected],individual["popularity"])])
        rng=np.random.default_rng(2026)
        bootstrap=np.array([rng.choice(difference,len(difference),replace=True).mean() for _ in range(1000)])
        ci=np.quantile(bootstrap,[.025,.975])
        winner=models[selected]
        # Diagnostic: all held-out positives plus 100 sampled unseen nonrelevant items.
        sampled=[]
        for user in users:
            user=int(user)
            row=winner.scores([user])[0]
            truth=data["test"][user]
            candidates=np.flatnonzero(np.isfinite(row))
            negatives=np.setdiff1d(candidates,truth,assume_unique=True)
            negatives=rng.choice(negatives,min(100,len(negatives)),replace=False)
            candidate_set=np.r_[truth,negatives]
            order=np.lexsort((candidate_set,-row[candidate_set]))[:10]
            sampled.append(ranking_metrics(candidate_set[order],truth))
        sampled_mean={key:float(np.mean([row[key] for row in sampled])) for key in sampled[0]}
        probes=np.random.default_rng(42).choice(users,1000,replace=False)
        baseline=winner.recommend_reference(probes)
        optimized=winner.recommend_batch(probes)
        exact_agreement=float(np.all(baseline==optimized,axis=1).mean())
        if exact_agreement<1:
            raise RuntimeError(f"Optimized rankings differ from the deterministic reference: {exact_agreement}")
        score_error=0.0
        for start in range(0,len(probes),64):
            batch=probes[start:start+64]
            batched=winner.scores(batch)
            scalar=np.vstack([winner.scores([user])[0] for user in batch])
            finite=np.isfinite(batched)&np.isfinite(scalar)
            score_error=max(score_error,float(np.max(np.abs(batched[finite]-scalar[finite]))))
        timings={"reference":[],"optimized":[]}
        for repeat in range(7):
            order=["reference","optimized"] if repeat%2==0 else ["optimized","reference"]
            for method in order:
                start=perf_counter()
                if method=="reference":
                    winner.recommend_reference(probes)
                else:
                    winner.recommend_batch(probes)
                timings[method].append((perf_counter()-start)*1000)
        performance={method:{"median_ms":float(np.median(values)),"p95_ms":float(np.percentile(values,95)),
                             "raw_ms":values} for method,values in timings.items()}
        performance.update({"queries":len(probes),"k":10,"batch_size":64,"repetitions":7,
                            "cpu_threads":2,"top10_list_agreement":exact_agreement,
                            "maximum_finite_score_difference":score_error,
                            "median_speedup":performance["reference"]["median_ms"]/performance["optimized"]["median_ms"],
                            "score_matrix_bytes_per_full_batch":64*B.shape[1]*8,
                            "scope":"Warm-process 1000-user scoring and ranking workload; excludes loading, HTTP, and concurrent production traffic"})
    rows=[]
    for name in models:
        rows.append({"method":name,"fit_seconds":fit_times[name],
                     **{"validation_"+key:value for key,value in validation[name].items()},
                     **{"test_"+key:value for key,value in test[name].items()}})
    write_csv(artifact/"model_comparison.csv",rows)
    # Serialize locally only. The public repository excludes data and profile matrices.
    bundle={"ranker":models[selected],"user_ids":data["users"],"item_ids":data["item_ids"],
            "titles":data["titles"],"dataset_sha256":data["info"]["archive_sha256"]}
    joblib.dump(bundle,artifact/"model.joblib",compress=3)
    report={"data":data["info"],"selected_model":selected,"validation":validation,"test":test,
            "paired_ndcg_vs_popularity":{"mean_difference":float(difference.mean()),
                                         "ci95_low":float(ci[0]),"ci95_high":float(ci[1]),
                                         "bootstrap_repetitions":1000,
                                         "scope":"Paired user resampling conditional on fitted models and this dataset"},
            "sampled_negative_diagnostic":{"negatives_per_user":100,"metrics":sampled_mean,
                                           "scope":"Diagnostic protocol; not headline full-catalogue results"},
            "performance":performance,"fit_seconds":fit_times,"als_objectives":objectives,
            "versions":{name:metadata.version(name) for name in ["numpy","scipy","joblib","threadpoolctl"]}}
    (artifact/"metrics.json").write_text(json.dumps(report,indent=2))
    print("RESULT",json.dumps({"selected":selected,"test":test[selected],"performance":performance}),flush=True)


if __name__=="__main__":
    main()
