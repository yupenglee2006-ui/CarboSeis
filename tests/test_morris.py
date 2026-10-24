"""Morris estimator: linear response recovered exactly; failed runs (NaN) are skipped."""
import numpy as np

from carboseis.analysis import morris_effects, morris_trajectories


def test_linear_model_and_failed_runs():
    k, r = 4, 6
    trajs, delta = morris_trajectories(k, r, seed=3)
    b = np.array([1.0, -2.0, 0.0, 0.5])
    Y = [np.column_stack([X @ b, (X @ b) ** 2]) for X in trajs]
    mu, sd, n = morris_effects(trajs, Y, delta, return_counts=True)
    assert np.allclose(mu[:, 0], np.abs(b) * delta)          # change over one Morris step
    assert np.allclose(sd[:, 0], 0.0) and (n == r).all()
    Y[0] = Y[0].copy()
    Y[0][2] = np.nan                                         # one failed evaluation
    mu2, _, n2 = morris_effects(trajs, Y, delta, return_counts=True)
    assert np.isfinite(mu2).all()
    assert n2.sum() == n.sum() - 2 * Y[0].shape[1]           # two effects lost per statistic
    assert np.allclose(mu2[:, 0], np.abs(b) * delta)
