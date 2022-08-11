import numpy as np
import lap

loss_inf = 10000


def greedy_assignment(score):
    """
    @param score:
    @return: matched indices and newborn det indices
    @egg: matched_indices, newborn_det_indices = greedy_assignment(cost_matrix_pred[0])
    """
    matched_indices = []
    if score.shape[1] == 0:
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    # match each tracker with largest-score detection
    for i in range(score.shape[0] - 1):
        j = score[i].argmax()
        if score[i][j] > 0.5:
            score[:-1, j] = -1
            matched_indices.append([i, j])
    matched_indices = np.array(matched_indices, np.int32).reshape(-1, 2)
    #  consider the unmatched detection as newborn if score > threshold
    newborn_det_indices = []
    for j in range(score.shape[1]):
        if j not in matched_indices[:, 1] and score[-1, j] > 0.5:
            newborn_det_indices.append(j)
    return matched_indices, newborn_det_indices


# This function is token from Toward-Realtime-MOT
def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return (np.empty((0, 2), dtype=int),
                tuple(range(cost_matrix.shape[0])),
                tuple(range(cost_matrix.shape[1])))
    matches, unmatched_a, unmatched_b = [], [], []
    virtual_tracker_line = cost_matrix[-1, :]
    print("virtual_tracker_line: ")
    print(virtual_tracker_line)
    num_det = cost_matrix.shape[1]
    num_track = cost_matrix.shape[0] - 1  # virtual tracker
    virtual_matrix = np.ones((num_det, num_det)) * loss_inf
    virtual_matrix[np.arange(num_det), np.arange(num_det)] = virtual_tracker_line
    cost_matrix_concat = np.vstack([cost_matrix[:-1, :], virtual_matrix])
    # cost, x, y = lap.lapjv(cost_matrix_concat, extend_cost=True, cost_limit=thresh)
    # print("=============================")
    cost, x, y = lap.lapjv(cost_matrix_concat, extend_cost=True)
    # print(cost)
    # print(x)
    # print(y)
    # print("*****************************")
    for idx in range(num_track):
        if x[idx] >= 0:
            matches.append([idx, x[idx]])
        else:
            unmatched_a.append(idx)
    for idx in range(num_det):
        if y[idx] == (num_track + idx):
            unmatched_b.append(idx)
    matches = np.asarray(matches)
    unmatched_b = np.asarray(unmatched_b)
    unmatched_a = np.array(unmatched_a)
    # print("matched:" + str(matches))
    # print("unmatched b:" + str(unmatched_b))
    # print("unmatched a:" + str(unmatched_a))
    return matches, unmatched_a, unmatched_b


