'''Compute BatchBALD uncertainty. Small modifications to ElementAI implementation at https://github.com/ElementAI/baal/blob/master/src/baal/active/heuristics/heuristics.py'''

import numpy as np
import scipy.stats
import torch
from scipy.special import xlogy
from torch import Tensor

reductions_map = {
    'max': lambda x: np.max(x, axis=tuple(range(1, x.ndim))),
    'min': lambda x: np.min(x, axis=tuple(range(1, x.ndim))),
    'mean': lambda x: np.mean(x, axis=tuple(range(1, x.ndim))),
    'sum': lambda x: np.sum(x, axis=tuple(range(1, x.ndim))),
    'none': lambda x: x,
}

def gather_expand(data, dim, index):
    """
    Gather indices `index` from `data` after expanding along dimension `dim`.
    Args:
        data (tensor): A tensor of data.
        dim (int): dimension to expand along.
        index (tensor): tensor with the indices to gather.
    References:
        Code from https://github.com/BlackHC/BatchBALD/blob/master/src/torch_utils.py
    Returns:
        Tensor with the same shape as `index`.
    """
    max_shape = [max(dr, ir) for dr, ir in zip(data.shape, index.shape)]
    new_data_shape = list(max_shape)
    new_data_shape[dim] = data.shape[dim]

    new_index_shape = list(max_shape)
    new_index_shape[dim] = index.shape[dim]

    data = data.expand(new_data_shape)
    index = index.expand(new_index_shape)

    return torch.gather(data, dim, index)


class BatchBALD:
    """
    Implementation of BatchBALD from https://github.com/BlackHC/BatchBALD
    Args:
        num_samples (int): Number of samples to select.
        num_draw (int): Number of draw to perform from the history.
                        From the paper `40000 // num_classes` is suggested.
        shuffle_prop (float): Amount of noise to put in the ranking. Helps with selection bias
            (default: 0.0).
        reduction (Union[str, callable]): function that aggregates the results
            (default: 'none').
    Notes:
        This implementation only returns the ranking and not the score.
    References:
        https://arxiv.org/abs/1906.08158
    Notes:
        K = iterations, C=classes
        Not tested on 4+ dims.
        """

    def __init__(self, num_samples=100, num_draw=500, shuffle_prop=0.0, reverse=True, reduction='none'):
        self.shuffle_prop = shuffle_prop
        self.reversed = reverse
        self.epsilon = 1e-5
        self.num_samples = num_samples
        self.num_draw = num_draw
        self.reduction = reduction if callable(reduction) else reductions_map[reduction]


    def _draw_choices(self, probs, n_choices):
        """
        Draw `n_choices` sample from `probs`.
        References:
            Code from https://github.com/BlackHC/BatchBALD/blob/master/src/torch_utils.py#L187
        Returns:
            choices: B... x `n_choices`
        """
        probs = probs.permute(0, 2, 1)
        probs_B_C = probs.reshape((-1, probs.shape[-1]))

        # samples: Ni... x draw_per_xx
        choices = torch.multinomial(probs_B_C,
                                    num_samples=n_choices, replacement=True)

        choices_b_M = choices.reshape(list(probs.shape[:-1]) + [n_choices])
        return choices_b_M.long()

    def _sample_from_history(self, probs, num_draw=1000):
        """
        Sample `num_draw` choices from `probs`
        Args:
            probs (Tensor[batch, classes, ..., iterations]): Tensor to be sampled from.
            num_draw (int): Number of draw.
        References:
            Code from https://github.com/BlackHC/BatchBALD/blob/master/src/joint_entropy/sampling.py
        Returns:
            Tensor[num_draw, iterations]
        """
        probs = torch.from_numpy(probs).double()

        n_iterations = probs.shape[-1]

        # [batch, draw, iterations]
        choices = self._draw_choices(probs, num_draw)

        # [batch, iterations, iterations, draw]
        expanded_choices_N_K_K_S = choices[:, None, :, :]
        expanded_probs_N_K_K_C = probs.permute(0, 2, 1)[:, :, None, :]

        probs = gather_expand(expanded_probs_N_K_K_C, dim=-1, index=expanded_choices_N_K_K_S)
        # exp sum log seems necessary to avoid 0s?
        entropies = torch.exp(torch.sum(torch.log(probs), dim=0, keepdim=False))
        entropies = entropies.reshape((n_iterations, -1))

        samples_M_K = entropies.t()
        return samples_M_K.numpy()

    def _conditional_entropy(self, probs):
        K = probs.shape[-1]
        return np.sum(-xlogy(probs, probs), axis=(1, -1)) / K

    def _joint_entropy(self, predictions, selected):
        """
        Compute the joint entropy between `preditions` and `selected`
        Args:
            predictions (Tensor): First tensor with shape [B, C, Iterations]
            selected (Tensor): Second tensor with shape [M, Iterations].
        References:
            Code from https://github.com/BlackHC/BatchBALD/blob/master/src/joint_entropy/sampling.py
        Notes:
            Only Classification is supported, not semantic segmentation or other.
        Returns:
            Generator yield B entropies.
        """
        K = predictions.shape[-1]
        C = predictions.shape[1]
        B = predictions.shape[0]
        M = selected.shape[0]
        predictions = predictions.swapaxes(1, 2)

        exp_y = np.array(
            [np.matmul(selected, predictions[i]) for i in range(predictions.shape[0])]) / K
        assert exp_y.shape == (B, M, C)
        mean_entropy = selected.mean(-1, keepdims=True)[None]
        assert mean_entropy.shape == (1, M, 1)

        step = 256
        for idx in range(0, exp_y.shape[0], step):
            b_preds = exp_y[idx:idx + step]
            yield np.sum(-xlogy(b_preds, b_preds) / mean_entropy, axis=(1, -1)) / M

    def _compute_bald(self, predictions):
        expected_entropy = - np.mean(np.sum(xlogy(predictions, predictions), axis=1), axis=-1)  # [batch size, ...]
        expected_p = np.mean(predictions, axis=-1)  # [batch_size, n_classes, ...]
        entropy_expected_p = - np.sum(xlogy(expected_p, expected_p),
                                      axis=1)  # [batch size, ...]
        bald_acq = entropy_expected_p - expected_entropy
        return bald_acq

    def _compute_batch_bald(self, predictions):
        MIN_SPREAD = 0.1
        COUNT = 0
        # Get conditional_entropies_B
        conditional_entropies_B = self._conditional_entropy(predictions)
        bald_out = self._compute_bald(predictions)
        # We start with the most uncertain sample according to BALD.
        history = self.reduction(bald_out).argsort()[-1:].tolist()
        history_unc = [bald_out[history[0]]]
        for step in range(self.num_samples):
            # Draw `num_draw` example from history, take entropy
            # TODO use numpy/numba
            selected = self._sample_from_history(predictions[history], num_draw=self.num_draw)

            # Compute join entropy
            joint_entropy = list(self._joint_entropy(predictions, selected))
            joint_entropy = np.concatenate(joint_entropy)

            partial_multi_bald_b = joint_entropy - conditional_entropies_B
            partial_multi_bald_b = self.reduction(partial_multi_bald_b)
            partial_multi_bald_b[..., np.array(history)] = -1000
            # Add best to history
            partial_multi_bald_b = partial_multi_bald_b.squeeze()
            assert partial_multi_bald_b.ndim == 1
            winner_index = partial_multi_bald_b.argmax()
            history.append(winner_index)
            history_unc.append(partial_multi_bald_b[winner_index])

            if partial_multi_bald_b.max() < MIN_SPREAD:
                COUNT += 1
                if COUNT > 50 or len(history) >= predictions.shape[0]:
                    break

        return np.array(history), np.array(history_unc)

    def get_uncertainties(self, predictions):
        """
        Get the uncertainties.
        Args:
            predictions (ndarray): Array of predictions with shape [batch_size, num_classes, num_iterations]
        Returns:
            Array of uncertainties
        """
        if isinstance(predictions, Tensor):
            predictions = predictions.numpy()
        ranks, scores = self._compute_batch_bald(predictions)
        scores = self.reduction(scores[:-1])
        if not np.all(np.isfinite(scores)):
            fixed = 0.0 if self.reversed else 10000
            warnings.warn(f"Invalid value in the score, will be put to {fixed}", UserWarning)
            scores[~np.isfinite(scores)] = fixed
        return scores
