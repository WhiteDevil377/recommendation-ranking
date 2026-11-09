"""Download MovieLens 1M locally and build documented per-user temporal holdouts."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import numpy as np
from scipy import sparse

ROOT=Path(__file__).resolve().parent
DATA_URL="https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def prepare():
    archive=ROOT/"data/ml-1m.zip"
    archive.parent.mkdir(exist_ok=True)
    if not archive.exists():
        with urlopen(DATA_URL,timeout=120) as response:
            archive.write_bytes(response.read())
    with ZipFile(archive) as z:
        ratings=np.loadtxt(BytesIO(z.read("ml-1m/ratings.dat").replace(b"::",b",")),delimiter=",",dtype=np.int64)
        movie_rows=[line.split("::",2) for line in z.read("ml-1m/movies.dat").decode("latin-1").splitlines()]
    if len(ratings)!=1000209:
        raise ValueError("Unexpected MovieLens 1M version")
    pairs=ratings[:,0]*10000+ratings[:,1]
    assert len(np.unique(pairs))==len(ratings),"Repeated user/item pairs"
    assert np.isin(ratings[:,2],[1,2,3,4,5]).all()
    movie_rows.sort(key=lambda row:int(row[0]))
    item_ids=np.array([int(row[0]) for row in movie_rows])
    users=np.unique(ratings[:,0])
    assert np.isin(ratings[:,1],item_ids).all()
    ratings=ratings[np.lexsort((ratings[:,1],ratings[:,3],ratings[:,0]))]
    boundaries=np.r_[0,np.flatnonzero(np.diff(ratings[:,0]))+1,len(ratings)]
    train_users,train_items,seen_users,seen_items=[],[],[],[]
    eligible=[]
    validation={}
    test={}
    cutoffs=[]
    for u,(start,end) in enumerate(zip(boundaries[:-1],boundaries[1:])):
        group=ratings[start:end]
        positives=group[group[:,2]>=4]
        cutoff=None
        if len(positives)>=20:
            tv=positives[int(len(positives)*.8),3]
            tt=positives[int(len(positives)*.9),3]
            before=positives[positives[:,3]<tv]
            val=positives[(positives[:,3]>=tv)&(positives[:,3]<tt)]
            later=positives[positives[:,3]>=tt]
            if len(before)>=5 and len(val)>0 and len(later)>0:
                assert before[:,3].max()<val[:,3].min() and val[:,3].max()<later[:,3].min()
                eligible.append(u)
                validation[u]=np.searchsorted(item_ids,val[:,1])
                test[u]=np.searchsorted(item_ids,later[:,1])
                cutoffs.append({"user_index":u,"train_latest":int(before[:,3].max()),
                                "validation_first":int(val[:,3].min()),"validation_last":int(val[:,3].max()),
                                "test_first":int(later[:,3].min())})
                positives=before
                cutoff=tv
        history=group if cutoff is None else group[group[:,3]<cutoff]
        train_users.extend([u]*len(positives))
        train_items.extend(np.searchsorted(item_ids,positives[:,1]).tolist())
        seen_users.extend([u]*len(history))
        seen_items.extend(np.searchsorted(item_ids,history[:,1]).tolist())
    shape=(len(users),len(item_ids))
    train=sparse.csr_matrix((np.ones(len(train_users)),(train_users,train_items)),shape=shape)
    seen=sparse.csr_matrix((np.ones(len(seen_users)),(seen_users,seen_items)),shape=shape)
    popularity=np.asarray(train.sum(axis=0)).ravel()
    cold_val=sum(int((popularity[items]==0).sum()) for items in validation.values())
    cold_test=sum(int((popularity[items]==0).sum()) for items in test.values())
    info={"source_url":DATA_URL,"archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
          "raw_ratings":len(ratings),"users":len(users),"catalogue_items":len(item_ids),
          "rated_items":len(np.unique(ratings[:,1])),"positive_threshold":4,
          "training_positive_events":int(train.nnz),"training_seen_events":int(seen.nnz),
          "evaluated_users":len(eligible),"users_excluded_from_ranking":len(users)-len(eligible),
          "validation_positive_events":sum(map(len,validation.values())),
          "test_positive_events":sum(map(len,test.values())),
          "cold_item_validation_events":cold_val,"cold_item_test_events":cold_test,
          "train_supported_items":int((popularity>0).sum()),
          "split":"Per-user positive timestamp 80/90 percentiles; ties assigned to the later block.",
          "eligibility":"At least 20 positives and nonempty strict train/validation/test blocks, with >=5 training positives.",
          "training_policy":"Ineligible users contribute all their positive feedback; eligible users contribute feedback before their validation cutoff.",
          "limitation":"This is a transductive per-user temporal benchmark, not a global chronological deployment simulation."}
    return {"train":train,"seen":seen,"users":users,"item_ids":item_ids,
            "titles":[row[1] for row in movie_rows],"eligible":np.array(eligible),
            "validation":validation,"test":test,"cutoffs":cutoffs,"info":info}


if __name__=="__main__":
    data=prepare()
    (ROOT/"artifacts").mkdir(exist_ok=True)
    (ROOT/"artifacts/data_quality.json").write_text(json.dumps(data["info"],indent=2))
    print(json.dumps(data["info"],indent=2))
