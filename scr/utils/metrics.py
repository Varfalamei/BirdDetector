import numpy as np
from sklearn.metrics import label_ranking_average_precision_score, average_precision_score


def padded_cmap(solution, submission, padding_factor=5):
    new_rows = np.ones((padding_factor, solution.shape[1]))
    padded_solution = np.concatenate((solution, new_rows))
    padded_submission = np.concatenate((submission, new_rows))
    score = average_precision_score(
        padded_solution.flatten(),
        padded_submission.flatten(),
        average='macro',
    )
    return score

def validation_epoch_end(y_true, predict):

    avg_RMAP = label_ranking_average_precision_score(y_true, predict)
    avg_custom = padded_cmap(y_true, predict)

    return {'val_RMAP': avg_RMAP, 'CMAP_5': avg_custom}