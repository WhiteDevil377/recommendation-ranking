"""Load the local recommendation bundle and score known or cold users."""
import argparse
import json
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from prepare import ROOT
from recommenders import partial_topk


class Service:
    def __init__(self):
        self.bundle=joblib.load(ROOT/"artifacts/model.joblib")
        self.lookup={int(user):i for i,user in enumerate(self.bundle["user_ids"])}

    def recommend(self,user_ids,k=10):
        if not isinstance(user_ids,list) or not 1<=len(user_ids)<=256:
            raise ValueError("Provide 1 to 256 user IDs")
        if any(isinstance(user,bool) or not isinstance(user,int) or user<1 for user in user_ids):
            raise ValueError("User IDs must be positive integers")
        if isinstance(k,bool) or not isinstance(k,int) or not 1<=k<=100:
            raise ValueError("k must be an integer from 1 to 100")
        ranker=self.bundle["ranker"]
        known_positions=[i for i,user in enumerate(user_ids) if user in self.lookup]
        known_indices=[self.lookup[user_ids[i]] for i in known_positions]
        with threadpool_limits(limits=2):
            rankings=ranker.recommend_batch(known_indices,k) if known_indices else []
        known={position:ranking for position,ranking in zip(known_positions,rankings)}
        fallback=partial_topk(ranker.popularity,k)
        result=[]
        for position,user in enumerate(user_ids):
            ranking=known.get(position,fallback)
            personalized=position in known and ranker.B.indptr[self.lookup[user]+1]>ranker.B.indptr[self.lookup[user]]
            result.append({"user_id":user,"mode":"personalized" if personalized else "popularity_fallback",
                           "recommendations":[{"movie_id":int(self.bundle["item_ids"][item]),
                                               "title":self.bundle["titles"][item]} for item in ranking]})
        return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id",type=int,required=True)
    parser.add_argument("--k",type=int,default=10)
    args=parser.parse_args()
    print(json.dumps(Service().recommend([args.user_id],args.k),indent=2))
