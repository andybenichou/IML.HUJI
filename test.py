import numpy as np

def _weighted_misclassification_error(values: np.ndarray, labels: np.ndarray) -> float:
    """
    Evaluate performance under misclassification loss function
    return sum of abs of labels, where sign(labels)!=sign(values).
    values are in (1,-1), labels don't have to be.
    Parameters
    ----------
    values: ndarray of shape (n_samples,)
        A feature vector to find a splitting threshold for
    labels: ndarray of shape (n_samples,)
        The labels to compare against
    """
    return np.abs(labels[np.sign(labels) != np.sign(values)]).sum()


if __name__ == '__main__':
    pass
    # sign = 1
    #
    # values = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    # labels = np.array([1, -1, 1, -1, -1, 1, 1, -1])
    #
    # print()
    # print("non sorted labels", labels)
    # print("non sorted values", values)
    #
    # print()
    # print("######      ANDY     #####")
    # print()
    #
    # sort_idx = np.argsort(values)
    # values, labels = values[sort_idx], labels[sort_idx]
    # losses = [np.sum(labels != sign)]
    #
    # print("values", values)
    # print()
    #
    # for label in labels:
    #     if sign == label:
    #         losses.append(losses[-1] + 1)
    #     else:
    #         losses.append(losses[-1] - 1)
    #
    # losses = np.array(losses) / len(labels)
    #
    # min_ind_loss = np.argmin(losses)
    #
    # print("losses", losses, "min_ind_loss", min_ind_loss)
    #
    # print("thr", values[min_ind_loss], "thr_err", losses[min_ind_loss])

    # print()
    # print("######      DANIEL     #####")
    # print()
    #
    # print("labels", labels)
    # print()
    #
    # lab = np.concatenate([[0], labels[:]])
    # print("lab", lab)
    #
    # lab2 = np.concatenate([labels[:], [0]])
    # print("lab2", lab2)
    # print()
    #
    # losses = np.minimum(np.cumsum(lab2 * sign),
    #                     np.cumsum(lab[::-1] * -sign)[::-1])
    # print('lab2 * sign', lab2 * sign)
    # print("np.cumsum(lab2 * sign)", np.cumsum(lab2 * sign))
    # print('lab[::-1] * -sign', lab[::-1] * -sign)
    # print('np.cumsum(lab[::-1] * -sign)[::-1]', np.cumsum(lab[::-1] * -sign)[::-1])
    # print('losses', losses)
    # print()
    #
    # min_id = np.argmin(losses)
    # print("min_id", min_id)
    #
    # if values[min_id] == np.max(values):
    #     print("thr", np.inf, "thr_err", losses[min_id])
    #
    # if values[min_id] == np.min(values):
    #     print("thr", -np.inf, "thr_err", losses[min_id])
    #
    # else:
    #     print("thr", values[min_id], "thr_err", losses[min_id])


    # print()
    # print("######      ALON     #####")
    # print()
    #
    # errors_ = [_weighted_misclassification_error(np.full((labels.size,), sign), labels)]
    #
    # print("losses_before", errors_)
    # print("labels", labels)
    # for i, threshold in enumerate(values[:-1]):
    #     errors_.append(errors_[-1] + sign * labels[i])
    #
    # errors_ = np.array(errors_)/len(values)
    #
    # print("losses_final", errors_)
    #
    # threshold_index_ = np.argmin(errors_)
    # print("min_ind", threshold_index_)
    # print()
    # print("thr", values[threshold_index_], "thr_err", errors_[threshold_index_])
    # print()

    # test = [(1, 2, 3, 4),
    #         (1, -1, 3, 5),
    #         (5, 6, -7, 8),
    #         (4, 8, 1, 4)]
    #
    # test = np.array(test)
    #
    # print(min(test, key=lambda el: el[2]))

    # print(values)
    # print(np.sign(values - values[threshold_index_]))
