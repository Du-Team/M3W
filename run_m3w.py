import argparse
import numpy as np
from M3W import border_tools as bt, BorderPeel, clustering_tools as ct

from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import SpectralEmbedding

parser = argparse.ArgumentParser(description='Multistep Three-way Clustering')
parser.add_argument('--input', type=str, metavar='<file path>', help='Path to comma separated input file',
                    required=True)
parser.add_argument('--output', type=str, metavar='<file path>', help='Path to output file', required=True)
parser.add_argument("--no-labels", help="Specify that input file has no ground truth labels", action="store_true")
parser.add_argument('--pca', type=int, metavar='<dimension>',
                    help='Perform dimensionality reduction using PCA to the given dimension before running the clustering',
                    required=False)
parser.add_argument('--spectral', type=int, metavar='<dimension>',
                    help='Perform sepctral embdding to the given dimension before running the clustering (If comibined with PCA, PCA is performed first)',
                    required=False)
args = parser.parse_args()
output_file_path = args.output
input_file_path = args.input
input_has_labels = not args.no_labels
pca_dim = args.pca
spectral_dim = args.spectral

debug_output_dir = None

# Parameters used for border peeling
k = 8
C = 1.6
T = 3
# Parameters used for three-way clustering
alpha = 0.6
beta = 0
# default values
border_precentile = 0.1
mean_border_eps = 0.15  # 0.15
stopping_precentile = 0.01

data, labels = ct.read_data(input_file_path, has_labels=input_has_labels)

min_cluster_size = 2

embeddings = data

if pca_dim is not None:
    if pca_dim >= len(embeddings[0]):
        print("PCA target dimension (%d) must be smaller than data dimension (%d)" % (pca_dim, len(embeddings[0])))
        exit(1)
    print("Performing PCA to %d dimensions" % pca_dim)
    pca = PCA(n_components=pca_dim)
    embeddings = pca.fit_transform(data)

if spectral_dim is not None:
    if spectral_dim >= len(embeddings[0]):
        print("Spectral Embedding dimension (%d) must be smaller than data dimension (%d)" % (
            spectral_dim, len(embeddings[0])))
        exit(1)
    print("Performing Spectral Embedding to %d dimensions" % spectral_dim)
    se = SpectralEmbedding(n_components=spectral_dim)
    embeddings = se.fit_transform(data)

print("Running Multistep Three-way Clustering on: %s" % input_file_path)
print("*" * 60)
lambda_estimate = bt.estimate_lambda(embeddings, k)
bp = BorderPeel.BorderPeel(
    mean_border_eps=mean_border_eps
    , max_iterations=T
    , k=k
    , plot_debug_output_dir=None
    , min_cluster_size=min_cluster_size
    , dist_threshold=lambda_estimate
    , convergence_constant=0
    , link_dist_expansion_factor=C
    , verbose=True
    , border_precentile=border_precentile
    , stopping_precentile=stopping_precentile
    , core_points_threshold=alpha
    , dvalue_threshold=beta
)

pred_membership = bp.fit_predict(embeddings)
clusters_count = pred_membership.shape[0]

print("*" * 60)
print("Found %d clusters" % clusters_count)
print("*" * 60)

clusters = []
nonzero_indices = np.nonzero(pred_membership)
nonzero_indices_rows = nonzero_indices[0]
nonzero_indices_cols = nonzero_indices[1]

with open(output_file_path, "w") as handle:
    for col in range(pred_membership.shape[1]):
        if col in nonzero_indices_cols:
            indices = nonzero_indices_cols == col
            clus = nonzero_indices_rows[indices]
        else:
            clus = np.array([-1])  # outliers
        clusters.append(np.amax(clus))
        # handle.write("%s\n" % ','.join(str(c) for c in clus)) # 输出所有标签
        handle.write("%s\n" % str(np.amax(clus)))  # 输出最终标签

print("Saved cluster results to %s" % output_file_path)
print("*" * 60)

# if input_has_labels:
# print("ARI: %0.3f" % adjusted_rand_score(clusters, labels))
# print("AMI: %0.3f" % adjusted_mutual_info_score(clusters, labels))
