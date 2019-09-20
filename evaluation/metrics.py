import torch

from data_utils import pose_features


def mean_error(errors, mode):
    if mode == 'squared':
        return (errors ** 2).mean(dim=1)
    elif mode == 'absolute':
        return errors.abs().mean(dim=1)
    else:
        raise ValueError('Unknown metric mode "{}"'.format(mode))


def max_error(errors, mode):
    if mode == 'squared':
        return (errors ** 2).max(dim=1).values
    elif mode == 'absolute':
        return errors.abs().max(dim=1).values
    else:
        raise ValueError('Unknown metric mode "{}"'.format(mode))


def mean_finger_error(errors, mode):
    errors_per_finger = pose_features.joints_of_all_fingers(errors)
    metric_per_finger = torch.zeros(errors.shape[0], 6)

    if mode == 'squared':
        metric_per_finger[:, 0] = errors_per_finger[:, 0, 0] ** 2
        metric_per_finger[:, 1:] = (errors_per_finger[:, 1:] ** 2).mean(dim=2)
    elif mode == 'absolute':
        metric_per_finger[:, 0] = errors_per_finger[:, 0, 0].abs()
        metric_per_finger[:, 1:] = errors_per_finger[:, :, 1:].abs().mean(dim=2)
    else:
        raise ValueError('Unknown metric mode "{}"'.format(mode))

    return metric_per_finger


# Basically the same as the "Percentage of Correct Keypoints" (PCK) metric but this term is not used
# here because a "keypoint" usually refers to a 2D concept. Furthermore some errors don't refer to
# points at all but rather to lengths or proportions.
def success_rate(errors, threshold):
    n_good_frames = (errors < threshold).sum()
    return n_good_frames / errors.shape[0]


def success_rate_curve(errors, thresholds):
    success_rates = [success_rate(errors, th) for th in thresholds]
    return success_rates


def area_under_curve(success_rates):
    return success_rates.mean()
