import hycoclip.COSNE.hyptorch.pmath as pmath
from hycoclip.COSNE.htsne_impl import TSNE as hTSNE
from sklearn.manifold import TSNE

def run_TSNE(embeddings, learning_rate = 1.0, learning_rate_for_h_loss = 0.0, perplexity=5, early_exaggeration=1, student_t_gamma=1.0):

    tsne = TSNE(n_components=2, method='exact', perplexity=perplexity, learning_rate=learning_rate, early_exaggeration=1)

    tsne_embeddings = tsne.fit_transform(embeddings)

    print ("\n\n")
    co_sne = hTSNE(n_components=2, verbose=0, method='exact', square_distances=True, 
                  metric='precomputed', learning_rate_for_h_loss=learning_rate_for_h_loss, student_t_gamma=student_t_gamma, learning_rate=learning_rate, n_iter=1000, perplexity=perplexity, early_exaggeration=early_exaggeration)

    dists = pmath.dist_matrix(embeddings, embeddings, c=1).numpy()


    CO_SNE_embedding = co_sne.fit_transform(dists, embeddings)


    _htsne = hTSNE(n_components=2, verbose=0, method='exact', square_distances=True, 
                  metric='precomputed', learning_rate_for_h_loss=0.0, student_t_gamma=1.0, learning_rate=learning_rate, n_iter=1000, perplexity=perplexity, early_exaggeration=early_exaggeration)

    HT_SNE_embeddings = _htsne.fit_transform(dists, embeddings)


    return tsne_embeddings, HT_SNE_embeddings, CO_SNE_embedding


def main(embeddings):

    return tsne_embeddings, HT_SNE_embeddings, CO_SNE_embedding