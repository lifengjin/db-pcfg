import matplotlib
matplotlib.use('Agg')
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.markers import MarkerStyle
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas, scipy.stats
import sys, os
# import matplotlib.colors as mcolors
#
# colors = [(1,0,0,c) for c in np.linspace(0,1,100)]
# cmapred = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=5)
# colors = [(0,0,1,c) for c in np.linspace(0,1,100)]
# cmapblue = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=5)
def symmetric_kl(p, q):
    kl1 = scipy.stats.entropy(p, q)
    kl2 = scipy.stats.entropy(q, p)
    return (kl1 + kl2) / 2

def limit_array(arr, arr_max, arr_min, r_limit=10):
    arr = (arr_max - arr_min) / 3 / (arr_max - arr + 1) * r_limit
    return arr

def burn_in_removal(arr, burn_in=10):
    return arr[burn_in:]

def plot_multiple_chains(sample_chains, logprobs, burn_in=100):
    """
    to plot a scatter plot of the posterior using dimensionality reduction techniques
    the distributions are first PCA-ed and then TSNE-ed into 2 dimensional data
    :param sample_chains: a list of numpy arrays of size (iters, params)
    :param logprobs: a list of logprobs corresponding to the samples (iters,)
    :return: a pdf file of the posterior
    """
    # preprocessing
    sample_chains = [burn_in_removal(x, burn_in) for x in sample_chains]
    max_logprobs = max([max(x[x != 0]) for x in logprobs])
    min_logprobs = min([min(x[x != 0]) for x in logprobs])
    print('max', max_logprobs, 'min', min_logprobs)
    logprobs = [limit_array(burn_in_removal(x, burn_in), max_logprobs, min_logprobs) for x in logprobs]

    num_sample_list = [x.shape[0] for x in sample_chains]
    data_matrix = np.vstack(sample_chains)
    # dim reduction
    # reduced_samples = PCA(n_components=200).fit_transform(data_matrix)
    bidigit_data_quad = PCA(n_components=4).fit_transform(data_matrix)
    # bidigit_data_tri = PCA(n_components=3).fit_transform(data_matrix)
    # bidigit_data_bi = PCA(n_components=2).fit_transform(data_matrix)
    # print(reduced_samples)
    # bidigit_data = reduced_samples

    # tsne_model_quad = TSNE(n_components=4, learning_rate=1000, metric='euclidean', perplexity=30)
    # tsne_model_tri = TSNE(n_components=3, learning_rate=1000, metric='euclidean', perplexity=30)
    # tsne_model_bi = TSNE(n_components=2, learning_rate=1000, metric='euclidean', perplexity=30)
    # tsne_model = TSNE(n_components=3, learning_rate=1000, metric=symmetric_kl, perplexity=50)
    # bidigit_data_quad = tsne_model_quad.fit_transform(bidigit_data)
    # bidigit_data_tri = tsne_model_tri.fit_transform(bidigit_data)
    # bidigit_data_bi = tsne_model_bi.fit_transform(bidigit_data)
    # print(bidigit_data)

    # generate the colors
    total_samples = sum(num_sample_list)
    num_chains = len(sample_chains)
    colors_per_point = np.zeros(total_samples)
    color_map = plt.get_cmap('rainbow')
    cur_index = 0
    running_sum = num_sample_list[0]
    all_markers = MarkerStyle.filled_markers
    markers = ['' for i in range(total_samples)]
    # chain_colors = []
    chain_markers = []
    chain_labels= []
    print('number of chains: ', num_chains)
    for i in range(total_samples):
        if i >= running_sum:
            cur_index += 1
            running_sum += num_sample_list[cur_index]
        if num_chains > 1:
            colors_per_point[i] = (cur_index+1) / num_chains  # one color per chain
        else:
            colors_per_point[i] = i / total_samples
        # colors_per_point[i] = (i - sum(num_sample_list[:cur_index])) / num_sample_list[cur_index] #all colors in a chain for temporal
        markers[i] = all_markers[cur_index]
        if markers[i] != markers[i-1] or not markers:
            # chain_colors.append(colors_per_point[i])
            chain_markers.append(markers[i])
            chain_labels.append('chain_'+str(cur_index))

    colors_per_point = ScalarMappable(cmap=color_map).to_rgba(colors_per_point)
    data_point_sizes = np.vstack(logprobs)
    for chain in data_point_sizes:
        print('max:', np.argmax(chain), max(chain))
    patches = []
    for i in range(len(chain_labels)):
        patches.append(Line2D(range(1), range(1), color='black',label=chain_labels[i], marker=chain_markers[i], markersize=2))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # num_sample_list.insert(0, 0)
    print(num_sample_list, chain_markers)
    if bidigit_data_quad[0, 0] > 0:
        bidigit_data_quad[:, 0] *= -1
    for i in range(num_chains):
        points = ax.scatter(bidigit_data_quad[sum(num_sample_list[:i]):sum(num_sample_list[:i+1]), 0], bidigit_data_quad[sum(num_sample_list[:i]):sum(num_sample_list[:i+1]), 1],
                            bidigit_data_quad[sum(num_sample_list[:i]):sum(num_sample_list[:i + 1]), 2], c=colors_per_point[sum(num_sample_list[:i]):sum(num_sample_list[:i+1])], s=data_point_sizes[i],
                            marker=chain_markers[i], cmap=color_map
                            , alpha=0.4, label='chain_'+str(i))
    ax.set_xlabel('1st Principle Component')
    ax.set_ylabel('2nd Principle Component')
    ax.set_zlabel('3rd Principle Component')
    # ax.legend(patches, chain_labels)
    ldg = ax.legend()
    for handle in ldg.legendHandles:
        handle._sizes = [30]
    pp = PdfPages('chains_posterior_3d' + '.pdf')
    fig.savefig(pp, format='pdf')
    pp.close()
    plt.cla()
    plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # num_sample_list.insert(0, 0)
    print(num_sample_list, chain_markers)
    for i in range(num_chains):
        points = ax.scatter(bidigit_data_quad[sum(num_sample_list[:i]):sum(num_sample_list[:i+1]), 1], bidigit_data_quad[sum(num_sample_list[:i]):sum(num_sample_list[:i+1]), 2],
                            bidigit_data_quad[sum(num_sample_list[:i]):sum(num_sample_list[:i + 1]), 3], c=colors_per_point[sum(num_sample_list[:i]):sum(num_sample_list[:i+1])], s=data_point_sizes[i],
                            marker=chain_markers[i], cmap=color_map
                            , alpha=0.4, label='chain_'+str(i))
    ax.set_xlabel('2nd Principle Component')
    ax.set_ylabel('3rd Principle Component')
    ax.set_zlabel('4th Principle Component')
    # ax.legend(patches, chain_labels)
    ldg = ax.legend()
    for handle in ldg.legendHandles:
        handle._sizes = [30]
    pp = PdfPages('chains_posterior_3d_notime' + '.pdf')
    fig.savefig(pp, format='pdf')
    pp.close()
    plt.cla()
    plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # num_sample_list.insert(0, 0)
    print(num_sample_list, chain_markers)
    for i in range(num_chains):
        points = ax.scatter(bidigit_data_quad[sum(num_sample_list[:i]):sum(num_sample_list[:i+1]), 0], bidigit_data_quad[sum(num_sample_list[:i]):sum(num_sample_list[:i+1]), 1],
                            c=colors_per_point[sum(num_sample_list[:i]):sum(num_sample_list[:i+1])], s=data_point_sizes[i],
                            marker=chain_markers[i], cmap=color_map
                            , alpha=0.4, label='chain_'+str(i))
    ax.set_xlabel('1st Principle Component')
    ax.set_ylabel('2nd Principle Component')
    # ax.legend(patches, chain_labels)
    ldg = ax.legend()
    for handle in ldg.legendHandles:
        handle._sizes = [30]
    pp = PdfPages('chains_posterior_2d' + '.pdf')
    fig.savefig(pp, format='pdf')
    pp.close()
    plt.cla()
    plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # num_sample_list.insert(0, 0)
    print(num_sample_list, chain_markers)
    for i in range(num_chains):
        points = ax.scatter(bidigit_data_quad[sum(num_sample_list[:i]):sum(num_sample_list[:i+1]), 1], bidigit_data_quad[sum(num_sample_list[:i]):sum(num_sample_list[:i+1]), 2],
                            c=colors_per_point[sum(num_sample_list[:i]):sum(num_sample_list[:i+1])], s=data_point_sizes[i],
                            marker=chain_markers[i], cmap=color_map
                            , alpha=0.4, label='chain_'+str(i))
    ax.set_xlabel('2nd Principle Component')
    ax.set_ylabel('3rd Principle Component')
    # ax.legend(patches, chain_labels)
    ldg = ax.legend()
    for handle in ldg.legendHandles:
        handle._sizes = [30]
    pp = PdfPages('chains_posterior_2d_notime' + '.pdf')
    fig.savefig(pp, format='pdf')
    pp.close()
    plt.cla()
    plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # num_sample_list.insert(0, 0)
    print(num_sample_list, chain_markers)
    for i in range(num_chains):
        points = ax.scatter(bidigit_data_quad[sum(num_sample_list[:i]):sum(num_sample_list[:i+1]), 1], bidigit_data_quad[sum(num_sample_list[:i]):sum(num_sample_list[:i+1]), 2],
                            # c=colors_per_point[sum(num_sample_list[:i]):sum(num_sample_list[:i+1])],
                            c = 'b',
                            s=np.ones_like(data_point_sizes[i]),
                            marker='o', cmap=color_map
                            , alpha=0.4, label='chain_'+str(i))
    ax.set_xlabel('2nd Principle Component')
    ax.set_ylabel('3rd Principle Component')
    # ax.legend(patches, chain_labels)
    ldg = ax.legend()
    for handle in ldg.legendHandles:
        handle._sizes = [30]
    pp = PdfPages('chains_posterior_2d_nosize_notime' + '.pdf')
    fig.savefig(pp, format='pdf')
    pp.close()
    plt.cla()
    plt.clf()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_trisurf(bidigit_data_bi[:, 0], bidigit_data_bi[:, 1], data_point_sizes.reshape(-1))
    # pp = PdfPages('chains_surface' + '.pdf')
    # fig.savefig(pp, format='pdf')
    # pp.close()
    # plt.cla()
    # plt.clf()

if __name__ == '__main__':
    # chains = [np.random.normal(5, 1, (25, 1000)), np.random.normal(-1, 2, (25, 1000))]
    # print(chains)
    working_dir = sys.argv[1]
    working_dirs = []
    if len(sys.argv[1:]) > 1:
        working_dirs = [working_dir+str(i) for i in range(int(sys.argv[2]))]  # if multiple directories then provide
        # the final index as second argument
    else:
        working_dirs.append(working_dir)
    chains = []
    logprobs = []
    for working_dir in working_dirs:
        nonterms = pandas.read_table(os.path.join(working_dir,'pcfg_nonterms.txt')).as_matrix()
        terms = pandas.read_table(os.path.join(working_dir,'pcfg_terms.txt')).as_matrix()
        allterms = np.hstack((terms,nonterms))
        chains.append(allterms)
        logprobs.append(pandas.read_table(os.path.join(working_dir,'pcfg_hypparams.txt')).logprob.as_matrix())
    plot_multiple_chains(chains, logprobs, burn_in=10)