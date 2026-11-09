"""Check ALS mathematics, temporal isolation, exact top-k, filtering, and the API."""
import json
import math
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request,urlopen
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from prepare import ROOT,prepare
from recommend import Service
from recommenders import (als_step,fit_als,full_sort_topk,partial_topk,ranking_metrics,
                           weighted_objective)


def main():
    B=sparse.csr_matrix([[1.,0.,1.,0.],[0.,1.,1.,0.],[0.,0.,0.,1.]])
    X=np.array([[.2,.1],[.3,.4],[.1,.5]])
    Y=np.array([[.1,.3],[.4,.2],[.5,.1],[.2,.2]])
    alpha,regularization=3.,.2
    C=1+alpha*B.toarray()
    dense=float((C*(B.toarray()-X@Y.T)**2).sum()+regularization*((X*X).sum()+(Y*Y).sum()))
    np.testing.assert_allclose(weighted_objective(B,X,Y,alpha,regularization),dense,rtol=1e-12)
    actual=als_step(B,Y,alpha,regularization)
    for u in range(B.shape[0]):
        expected=np.linalg.solve(Y.T@np.diag(C[u])@Y+regularization*np.eye(2),Y.T@(C[u]*B.toarray()[u]))
        np.testing.assert_allclose(actual[u],expected,rtol=1e-12,atol=1e-12)
    with threadpool_limits(limits=2):
        _,_,losses=fit_als(B,factors=2,alpha=3,regularization=.2,iterations=3)
    assert all(b<=a+1e-8 for a,b in zip(losses,losses[1:]))
    score_cases=[np.array([2.,2.,2.,1.,0.]),np.array([-np.inf,0.,0.,1.,1.]),
                 np.random.default_rng(42).normal(size=100)]
    for scores in score_cases:
        for k in [1,3]:
            np.testing.assert_array_equal(partial_topk(scores,k),full_sort_topk(scores,k))
    metric=ranking_metrics([3,8,4],[3,4])
    np.testing.assert_allclose(metric["ndcg_at_10"],1.5/(1+1/math.log2(3)))
    assert metric["recall_at_10"]==1 and metric["mrr_at_10"]==1
    data=prepare()
    for cutoff in data["cutoffs"]:
        assert cutoff["train_latest"]<cutoff["validation_first"]<=cutoff["validation_last"]<cutoff["test_first"]
    for u in data["eligible"]:
        seen=set(data["seen"].indices[data["seen"].indptr[u]:data["seen"].indptr[u+1]])
        assert not seen&set(data["validation"][int(u)])
        assert not seen&set(data["test"][int(u)])
        assert not set(data["validation"][int(u)])&set(data["test"][int(u)])
    service=Service()
    model=service.bundle["ranker"]
    probes=np.random.default_rng(9).choice(data["eligible"],100,replace=False)
    with threadpool_limits(limits=2):
        reference=model.recommend_reference(probes)
        fast=model.recommend_batch(probes)
    np.testing.assert_array_equal(reference,fast)
    for u,ranking in zip(probes,fast):
        seen=set(data["seen"].indices[data["seen"].indptr[u]:data["seen"].indptr[u+1]])
        assert len(set(ranking))==10 and not seen&set(ranking)
    known=int(data["users"][data["eligible"][0]])
    result=service.recommend([known,999999],5)
    assert len(result)==2 and len(result[0]["recommendations"])==5
    assert result[1]["mode"]=="popularity_fallback"
    for ids,k in [([],10),([True],10),([1],0),([0],10),(["1"],10)]:
        try:
            service.recommend(ids,k)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input accepted")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0))
        port=sock.getsockname()[1]
    process=subprocess.Popen([sys.executable,str(ROOT/"serve.py"),"--port",str(port)],
                             cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    base=f"http://127.0.0.1:{port}"
    try:
        ready=False
        for _ in range(80):
            if process.poll() is not None:
                raise RuntimeError(process.stderr.read().decode())
            try:
                with urlopen(base+"/health",timeout=.5) as response:
                    assert json.load(response)["status"]=="ok"
                ready=True
                break
            except OSError:
                time.sleep(.25)
        assert ready
        with urlopen(base+f"/recommend?user_id={known}&k=3",timeout=10) as response:
            assert len(json.load(response)["results"][0]["recommendations"])==3
        request=Request(base+"/recommend",data=json.dumps({"user_ids":[known,999999],"k":3}).encode(),
                        headers={"Content-Type":"application/json"},method="POST")
        with urlopen(request,timeout=10) as response:
            assert len(json.load(response)["results"])==2
        try:
            urlopen(base+"/recommend?user_id=0",timeout=5)
        except HTTPError as error:
            assert error.code==400
        else:
            raise AssertionError("API accepted invalid user")
    finally:
        process.terminate()
        process.communicate(timeout=10)
    print("PASS: dense ALS loss and normal equations, objective descent, exact tie-aware top-k, hand ranking metric, per-user temporal isolation, seen-item filtering, cold fallback, batch API and invalid input")


if __name__=="__main__":
    main()
