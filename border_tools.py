from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsRegressor
from scipy.interpolate import griddata
from python_algorithms.basic import union_find
from time import time
import numpy as np
import copy


def rknn_with_distance_transform(data, k, transform):
    rows_count = len(data)
    k = min(k, rows_count - 1)
    rknn_values = np.zeros(rows_count)
    nbrs = NearestNeighbors(n_neighbors=k).fit(data)
    distances, indices = nbrs.kneighbors()

    for index, indRow, distRow in zip(range(len(indices)), indices, distances):
        for i, d in zip(indRow, distRow):
            transform(rknn_values, index, i, d, indices, distances, k)
    return rknn_values, nbrs


def exp_local_scaling_transform(rknnValues, first_index, second_index, dist, indices, distances, k):
    first_scale_index = k
    if len(distances[first_index]) <= first_scale_index:
        first_scale_index = len(distances[first_index]) - 1
    local_sigma = distances[first_index][first_scale_index]
    rknnValues[second_index] = rknnValues[second_index] + np.exp(-(dist * dist) / (local_sigma * local_sigma))


def border_peel_single(data, border_func, threshold_func, precentile=0.1, verbose=False):
    border_values, nbrs = border_func(data)
    if border_values is None:
        return None, None, None
    # calculate the precentile of the border value..
    if precentile > 0:
        sorted = np.array(border_values)
        sorted.sort()
        index_prcentile = int(len(border_values) * precentile)
        threshold_value = sorted[index_prcentile]
        if verbose:
            print("threshold value %0.3f for precentile: %0.3f" % (threshold_value, precentile))
        filter = border_values > threshold_value
    else:
        filter = threshold_func(border_values)
    return filter, border_values, nbrs


def mat_to_1d(arr):
    if hasattr(arr, 'A1'):
        return arr.A1
    return arr


def evaluateLinkThresholds(data, filter, nbrs, dist_threshold):
    dataLength = len(data)
    xy = []
    Z = []

    distances, indices = nbrs.kneighbors()
    for index, indRow, distRow in zip(range(dataLength), indices, distances):
        if (not filter[index]):
            continue

        # look for the nearest neighbor whose isn't border
        for j, d in zip(indRow[1:], distRow[1:]):
            if (not filter[j]):
                xy.append(mat_to_1d(data[j]).tolist())
                Z.append(d)
                break

    # todo: using nearest method here in order to avoid getting nans..should also try
    # and see if using linear is better..
    if (len(Z) == 0):
        return None

    thresholds = griddata(np.matrix(xy), np.array(Z), data, method='nearest')
    # thresholds = griddata(np.matrix(xy), np.array(Z), data, method='linear')

    return thresholds


def update_link_thresholds(
        current_data, original_indices, original_data, thresholds, dist_threshold, link_dist_expansion_factor, k=10
):
    # index the filters according to the original data indices:
    original_data_filter = np.zeros(len(original_data)).astype(int)
    original_data_filter[original_indices] = 1

    xy = original_data[(original_data_filter == 0)]  # cords of borders
    Z = thresholds[(original_data_filter == 0)]  # link thresholds of borders

    knn = KNeighborsRegressor(k, weights="uniform")

    try:
        new_thresholds = knn.fit(xy, Z).predict(current_data)
    except:
        print("failed to run kneighbours regressor")
        return thresholds

    for i, p, t in zip(range(len(current_data)), current_data, new_thresholds):
        original_index = original_indices[i]
        if np.isnan(t):
            print("threshold is nan")

        if np.isnan(t) or (t * link_dist_expansion_factor) > dist_threshold:
            # print "setting dist threshold"
            thresholds[original_index] = dist_threshold
        else:
            # print "setting threhold: %.2f"%(t * link_dist_expansion_factor)
            thresholds[original_index] = t * link_dist_expansion_factor

    return thresholds


class StopWatch:
    def __init__(self):
        self.time = time()

    def t(self, message):
        print("watch: %s: %0.4f" % (message, time() - self.time))

        # pass
        # self.time = time()


def dynamic_3w(
        data
        , border_func
        , threshold_func
        , max_iterations=150
        , min_iterations=3
        , mean_border_eps=-1
        , plot_debug_output_dir=None
        , min_cluster_size=3
        , dist_threshold=3
        , convergence_constant=0
        , link_dist_expansion_factor=3
        , k=10
        , verbose=True
        , precentile=0.1
        , vis_data=None
        , stopping_precentile=0
        , should_merge_core_points=True
        , debug_marker_size=70
        , core_points_threshold=1
        , dvalue_threshold=1
):
    watch = StopWatch()

    # original_data_points_indices = {}
    data_length = len(data)
    cluster_uf = union_find.UF(data_length)
    original_indices = np.arange(data_length)
    link_thresholds = np.ones(data_length) * dist_threshold
    # for d,i in zip(data,xrange(data_length)):
    #    original_data_points_indices[tuple(mat_to_1d(d))] = i

    if vis_data is None:
        vis_data = data
    original_vis_data = vis_data
    current_vis_data = vis_data

    original_data = data
    current_data = data
    data = None

    # 1 if the point wasn't peeled yet, 0 if it was

    # original_data_filter  = np.ones(data_length)

    # plt_dbg_session = DebugPlotSession(plot_debug_output_dir, marker_size=debug_marker_size, line_width=1.0)
    initial_core_points = []
    initial_core_points_original_indices = []
    data_sets = [original_data]
    nbrs = NearestNeighbors(n_neighbors=len(current_data) - 1).fit(current_data)
    nbrs_distances, nbrs_indices = nbrs.kneighbors()
    max_core_points = stopping_precentile * data_length
    border_values_per_iteration = []
    border_indices_per_iteration = []

    if mean_border_eps > 0:
        mean_border_vals = []

    watch.t("initialization")

    for t in range(max_iterations):
        start_time = time()
        filter, border_values, nbrs = border_peel_single(
            current_data
            , border_func
            , threshold_func
            , precentile=precentile
            , verbose=verbose
        )
        watch.t("rknn")
        peeled_border_values = border_values[filter == False]
        border_values_per_iteration.append(peeled_border_values)

        # # iteration termination condition
        # if mean_border_eps > 0:
        #     mean_border_vals.append(np.mean(peeled_border_values))
        #     if t >= min_iterations and len(mean_border_vals) > 2:
        #         ratio_diff = (mean_border_vals[-1] / mean_border_vals[-2]) - (
        #                 mean_border_vals[-2] / mean_border_vals[-3])
        #         if verbose:
        #             print("mean border ratio difference: %0.3f" % (ratio_diff))
        #         if ratio_diff > mean_border_eps:
        #             if verbose:
        #                 print("mean border ratio is larger than set value, stopping peeling")
        #             break

        if nbrs is None:
            if verbose:
                print("nbrs are none, breaking")
            break

        watch.t("mean borders")
        # filter out data points
        links = []

        # mark and distinguish all core (=1) and border (=0) points
        original_data_filter = np.zeros(data_length).astype(int)
        original_indices_new = original_indices[filter]
        border_indices_new = original_indices[~filter]  # 本次迭代剥离的边界点在原数据集中的位置索引
        border_indices_per_iteration.append(border_indices_new)
        original_data_filter[original_indices_new] = 1

        watch.t("nearset neighbors")

        # update link thresholds of borders
        for d, i, nn_inds, nn_dists in zip(current_data, range(len(current_data)), nbrs_indices, nbrs_distances):
            # skip non border points
            if filter[i]:
                continue
            # find the next neighbor we can link to
            original_index = original_indices[i]
            link_nig_index = -1
            link_nig_dist = -1
            # make sure we exclude self point here..
            link_threshold = link_thresholds[original_index]
            for nig_index, nig_dist in zip(nn_inds, nn_dists):
                if nig_dist > link_threshold:
                    break

                if not original_data_filter[nig_index]:  # choose non border and skip border
                    continue

                link_nig_index = nig_index
                link_nig_dist = nig_dist
                break  # nearst non border of the border

            # do not link this point to any other point (but still remove it), consider it as noise instead for now
            # this will generally mean that this point is sorrounded by other border points
            if link_nig_index > -1:
                links.append((i, link_nig_index))
                original_link_nig_index = link_nig_index
                # cluster_uf.union(original_index, original_link_nig_index)  # border link to core
                link_thresholds[original_index] = link_nig_dist
            # else:  # leave it in a seperate cluster...
            #     initial_core_points.append(d) # add outliers
            #     initial_core_points_original_indices.append(original_index)
            #     filter[i] = False

        watch.t("association")

        # if (plot_debug_output_dir != None):
        #     original_data_filter = 2 * np.ones(len(original_data)).astype(int)
        #     for i,p,f in zip(range(len(current_data)), current_data,filter):
        #         #original_index = original_data_points_indices[tuple(p)]
        #         original_index = original_indices[i]
        #         original_data_filter[original_index] = f
        #
        #     plt_dbg_session.plot_and_save(original_vis_data, original_data_filter)

        # update params for next iterations
        # interpolate the threshold values for the next iteration:
        previous_iteration_data_length = len(current_data)
        # filter the data:
        current_data = current_data[filter]
        current_vis_data = current_vis_data[filter]
        data_sets.append(current_data)
        original_indices = original_indices_new
        nbrs_indices = nbrs_indices[filter]
        nbrs_distances = nbrs_distances[filter]

        watch.t("filter")

        # calculate and update the link thresholds for non borders:
        link_thresholds = update_link_thresholds(
            current_data
            , original_indices
            , original_data
            , link_thresholds
            , dist_threshold
            , link_dist_expansion_factor
            , k=k
        )

        watch.t("thresholds")

        if verbose:
            print("iteration %d, peeled: %d, remaining data points: %d, number of sets: %d" \
                  % (t, abs(len(current_data) - previous_iteration_data_length), len(current_data), cluster_uf.count()))
        if abs(len(current_data) - previous_iteration_data_length) < convergence_constant:
            if verbose:
                print("stopping peeling since difference between remaining data points and current is: %d" % (
                    abs(len(current_data) - previous_iteration_data_length)))
            break

        if max_core_points > len(current_data):
            if verbose:
                print("number of core points is below the max threshold, stopping")
            break

    watch.t("before merge")
    clusters = np.ones(len(original_data)) * -1

    if verbose:
        print("before merge: %d" % cluster_uf.count())
    print(initial_core_points_original_indices)

    # core region clustering
    core_points_merged = current_data.tolist() + initial_core_points

    original_core_points_indices = original_indices.tolist() + initial_core_points_original_indices

    core_points = np.ndarray(shape=(len(core_points_merged), len(core_points_merged[0])),
                             buffer=np.array(core_points_merged))

    watch.t("before to associations map")

    uf_map = uf_to_associations_map(cluster_uf, core_points, original_core_points_indices)

    watch.t("after associations map")

    non_merged_core_points = copy.deepcopy(core_points)

    if should_merge_core_points:
        merge_core_points(core_points, link_thresholds, original_core_points_indices, cluster_uf, verbose)

    watch.t("core points merge")

    if verbose:
        print("after merge: %d" % cluster_uf.count())

    cluster_lists = union_find_to_lists(cluster_uf)

    cluster_index = 0

    for l in cluster_lists:
        if len(l) < min_cluster_size:
            continue

        for i in l:
            clusters[i] = cluster_index

        cluster_index += 1

    # construct membership array
    membership = np.zeros((cluster_index, len(original_data)))
    for col in range(len(clusters)):
        row = int(clusters[col])
        if row != -1:
            membership[row][col] = 1

    # membership array of cores
    clusters_core = clusters[original_core_points_indices]
    membership_core = membership[:, original_core_points_indices]

    # border region clustering
    for i in range(len(border_indices_per_iteration) - 1, -1, -1):
        border_indices = border_indices_per_iteration[i]  # Index of boundary points in the innermost layer
        border_data = original_data[border_indices, :]
        core_data = original_data[original_core_points_indices, :]
        # find the set of core k nearest neighbors of x_i
        nbrs_core = NearestNeighbors(n_neighbors=k).fit(core_data)
        nbrs_core_distances, nbrs_core_indices = nbrs_core.kneighbors(border_data)
        for j in range(len(border_indices)):
            # assign peeled points to the cluster whose k-core neighbors have the most membership
            nei_members = membership_core[:, nbrs_core_indices[j, :]]
            nei_mean_members = np.mean(nei_members, axis=1)
            nei_most_cluster = np.argmax(nei_mean_members)
            nei_most_pos = nei_mean_members[nei_most_cluster]
            membership[nei_most_cluster, border_indices[j]] = 1
            if nei_most_pos >= core_points_threshold or (np.count_nonzero(
                    ((nei_most_pos - nei_mean_members) <= (dvalue_threshold / cluster_index))) <= 1):
                # assign the peeled point to a core region
                clusters_core = np.append(clusters_core, nei_most_cluster)
                add_membership_core = np.zeros((cluster_index, 1))
                add_membership_core[nei_most_cluster] = 1
                membership_core = np.concatenate((membership_core, add_membership_core), axis=1)
                original_core_points_indices.append(border_indices[j])
            else:  # assign the peeled point to border regions
                regions_filter = (nei_most_pos - nei_mean_members) <= (dvalue_threshold / cluster_index)
                nei_clusters = np.where(regions_filter == True)
                membership[nei_clusters, border_indices[j]] = 1

    # if plot_debug_output_dir != None:
    #     for original_index in original_indices:
    #         core_clusters[original_index] = clusters[original_index]
    #
    #     # draw only core points clusters
    #     plt_dbg_session.plot_clusters_and_save(original_vis_data, core_clusters, noise_data_color = 'white')
    #
    #     # draw all of the clusters
    #     plt_dbg_session.plot_clusters_and_save(original_vis_data, clusters)

    watch.t("before return")

    return membership, core_points, non_merged_core_points, data_sets, uf_map, link_thresholds, \
           border_values_per_iteration, original_indices


def estimate_lambda(data, k):
    '''
    Parameters
    ----------
    data
    k

    Returns
    -------

    '''
    nbrs = NearestNeighbors(n_neighbors=k).fit(data)
    distances, indices = nbrs.kneighbors()

    all_dists = distances.flatten()
    return np.mean(all_dists) + np.std(all_dists)


def union_find_to_lists(uf):
    list_lists = []
    reps_to_sets = {}

    for i in range(len(uf._id)):
        r = uf.find(i)
        if r not in reps_to_sets:
            reps_to_sets[r] = len(list_lists)
            list_lists.append([i])
        else:
            list_lists[reps_to_sets[r]].append(i)

    return list_lists


def uf_to_associations_map(uf, core_points, original_indices):
    reps_items = {}
    reps_to_core = {}

    for original_index in original_indices:
        r = uf.find(original_index)
        reps_to_core[r] = original_index

    for i in range(len(uf._id)):
        r = uf.find(i)

        if r not in reps_to_core:
            reps_to_core[r] = i

        k = reps_to_core[r]
        if k not in reps_items:
            reps_items[k] = []
        reps_items[k].append(i)

    return reps_items


def merge_core_points(core_points, link_thresholds, original_indices, cluster_sets, verbose=False):
    t = StopWatch()
    print(original_indices)
    try:
        nbrs = NearestNeighbors(n_neighbors=len(core_points) - 1).fit(core_points, core_points)
        distances, indices = nbrs.kneighbors()
    except Exception as err:
        if verbose:
            print("faiiled to find nearest neighbors for core points")
            print(err)
        return
    t.t("Core points - after nn")
    for original_index, ind_row, dist_row in zip(original_indices, indices, distances):
        link_threshold = link_thresholds[original_index]
        for i, d in zip(ind_row[1:], dist_row[1:]):
            if d > link_threshold:
                break
            n_original_index = original_indices[i]
            cluster_sets.union(original_index, n_original_index)
    t.t("Core points - after merge")


def border_peel_rknn_exp_transform_local(data, k, threshold, iterations, debug_output_dir=None,
                                         dist_threshold=3, link_dist_expansion_factor=3, precentile=0, verbose=True):
    border_func = lambda data: rknn_with_distance_transform(data, k, exp_local_scaling_transform)
    threshold_func = lambda value: value > threshold
    return dynamic_3w(data, iterations, border_func, threshold_func,
                      plot_debug_output_dir=debug_output_dir, k=k, precentile=precentile,
                      dist_threshold=dist_threshold, link_dist_expansion_factor=link_dist_expansion_factor,
                      verbose=verbose)
