"""Original implementations of popularity, cosine item-KNN, and weighted implicit ALS."""
import numpy as np
from scipy import sparse


def full_sort_topk(scores,k):
    indices=np.arange(len(scores))
    return np.lexsort((indices,-scores))[:k]


def partial_topk(scores,k):
    if not 1<=k<=len(scores):
        raise ValueError("Invalid k")
    subset=np.argpartition(-scores,k-1)[:k]
    boundary=scores[subset].min()
    above=np.flatnonzero(scores>boundary)
    ties=np.flatnonzero(scores==boundary)[:k-len(above)]
    selected=np.r_[above,ties]
    return selected[np.lexsort((selected,-scores[selected]))]


def item_similarity(B,neighbors,shrinkage=10):
    cooccurrence=(B.T@B).toarray()
    norms=np.sqrt(np.maximum(np.diag(cooccurrence),1))
    cooccurrence*=cooccurrence/(cooccurrence+shrinkage)
    cooccurrence/=norms[:,None]
    cooccurrence/=norms[None,:]
    np.fill_diagonal(cooccurrence,0)
    rows,columns,values=[],[],[]
    for item,row in enumerate(cooccurrence):
        indices=partial_topk(row,min(neighbors,len(row)))
        indices=indices[row[indices]>0]
        rows.extend([item]*len(indices))
        columns.extend(indices.tolist())
        values.extend(row[indices].tolist())
    return sparse.csr_matrix((values,(rows,columns)),shape=cooccurrence.shape)


def als_step(interactions,fixed,alpha,regularization):
    factors=fixed.shape[1]
    base=fixed.T@fixed+regularization*np.eye(factors)
    result=np.zeros((interactions.shape[0],factors),dtype=np.float64)
    for row in range(interactions.shape[0]):
        start,end=interactions.indptr[row:row+2]
        vectors=fixed[interactions.indices[start:end]]
        if len(vectors):
            A=base+alpha*(vectors.T@vectors)
            b=(1+alpha)*vectors.sum(axis=0)
            result[row]=np.linalg.solve(A,b)
    return result


def weighted_objective(B,X,Y,alpha,regularization):
    # Full-catalogue zero-preference terms via Gram matrices, plus observed corrections.
    loss=float(np.sum((X.T@X)*(Y.T@Y)))
    user_indices=np.repeat(np.arange(B.shape[0]),np.diff(B.indptr))
    for start in range(0,B.nnz,32768):
        a=user_indices[start:start+32768]
        b=B.indices[start:start+32768]
        scores=np.einsum("ij,ij->i",X[a],Y[b])
        loss+=float(((1+alpha)*(1-scores)**2-scores**2).sum())
    loss+=regularization*float((X*X).sum()+(Y*Y).sum())
    return loss


def fit_als(B,factors=32,alpha=20,regularization=.1,iterations=10,seed=42):
    rng=np.random.default_rng(seed)
    X=rng.normal(0,.01,size=(B.shape[0],factors))
    Y=rng.normal(0,.01,size=(B.shape[1],factors))
    transpose=B.T.tocsr()
    history=[weighted_objective(B,X,Y,alpha,regularization)]
    for iteration in range(iterations):
        X=als_step(B,Y,alpha,regularization)
        Y=als_step(transpose,X,alpha,regularization)
        loss=weighted_objective(B,X,Y,alpha,regularization)
        if loss>history[-1]*(1+1e-8):
            raise RuntimeError("ALS objective increased")
        history.append(loss)
        print("ALS",factors,alpha,"iteration",iteration+1,"objective",round(loss,2),flush=True)
    return X,Y,history


class Ranker:
    def __init__(self,name,B,seen,popularity,X=None,Y=None,similarity=None):
        self.name=name
        self.B=B
        self.seen=seen
        self.popularity=np.asarray(popularity,dtype=np.float64)
        self.X,self.Y=X,Y
        self.similarity=similarity

    def scores(self,users):
        users=np.asarray(users,dtype=int)
        if self.X is not None:
            scores=self.X[users]@self.Y.T
        elif self.similarity is not None:
            scores=(self.B[users]@self.similarity).toarray()
        else:
            scores=np.tile(self.popularity,(len(users),1))
        scores=np.asarray(scores,dtype=np.float64)
        empty=np.diff(self.B.indptr)[users]==0
        scores[empty]=self.popularity
        for local,user in enumerate(users):
            start,end=self.seen.indptr[user:user+2]
            scores[local,self.seen.indices[start:end]]=-np.inf
        return scores

    def recommend_batch(self,users,k=10,batch_size=64):
        users=np.asarray(users,dtype=int)
        out=[]
        for start in range(0,len(users),batch_size):
            scores=self.scores(users[start:start+batch_size])
            out.extend(partial_topk(row,k) for row in scores)
        return np.asarray(out)

    def recommend_reference(self,users,k=10):
        return np.asarray([full_sort_topk(self.scores([user])[0],k) for user in users])


def ranking_metrics(ranking,relevant,k=10):
    relevant=set(map(int,relevant))
    hits=np.array([int(item) in relevant for item in ranking[:k]],dtype=float)
    ideal=sum(1/np.log2(rank+2) for rank in range(min(len(relevant),k)))
    positions=np.flatnonzero(hits)
    return {"ndcg_at_10":float((hits/np.log2(np.arange(len(hits))+2)).sum()/ideal) if ideal else 0.0,
            "recall_at_10":float(hits.sum()/len(relevant)) if relevant else 0.0,
            "mrr_at_10":float(1/(positions[0]+1)) if len(positions) else 0.0}
