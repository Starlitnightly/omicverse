import numpy as np
import torch

from copy import deepcopy


class ModelSaver:
    """In-memory saver for model parameters.

    Storing weights in memory is faster than saving to disk with torch.save.
    """
    def __init__(self, model):
        self.model = model

    def save(self):
        self.state_dict = deepcopy(self.model.state_dict())

    def restore(self):
        self.model.load_state_dict(self.state_dict)


class EarlyStopping:
    """Base class for an early stopping monitor that says when it's time to stop training.

    Examples
    --------
    early_stopping = EarlyStopping()
    for epoch in range(max_epochs):
        sess.run(train_op)  # perform training operation
        early_stopping.next_step()
        if early_stopping.should_stop():
            break
        if early_stopping.should_save():
            model_saver.save()  # save model weights

    """
    def __init__(self):
        pass

    def reset(self):
        """Reset the internal state."""
        raise NotImplementedError

    def next_step(self):
        """Should be called at every iteration."""
        raise NotImplementedError

    def should_save(self):
        """Says if it's time to save model weights."""
        raise NotImplementedError

    def should_stop(self):
        """Says if it's time to stop training."""
        raise NotImplementedError


class NoEarlyStopping(EarlyStopping):
    """No early stopping."""
    def __init__(self):
        super().__init__()
        pass

    def reset(self):
        pass

    def next_step(self):
        pass

    def should_stop(self):
        return False

    def should_save(self):
        return False


class NoImprovementStopping(EarlyStopping):
    """Stop training when the validation metric stops improving.

    Parameters
    ----------
    validation_fn : function
        Calling this function returns the current value of the validation metric.
    mode : {'min', 'max'}
        Should the validation metric be minimized or maximized?
    patience : int
        Number of iterations without improvement before stopping.
    tolerance : float
        Minimal improvement in validation metric to not trigger patience.
    relative : bool
        Is tolerance measured in absolute units or relatively?

    Attributes
    ----------
    _best_value : float
        Best value of the validation loss.
    _num_bad_epochs : int
        Number of epochs since last significant improvement in validation metric.
    _time_to_save : bool
        Is it time to save the model weights?
    _is_better : function
        Tells if new validation metric value is better than the best one so far.
        Signature self._is_better(new_value, best_value).

    """
    def __init__(self, validation_fn, mode='min', patience=10, tolerance=0.0, relative=False):
        super().__init__()
        self.validation_fn = validation_fn
        self.mode = mode
        self.patience = patience
        self.tolerance = tolerance
        self.relative = relative
        self.reset()

        if mode not in ['min', 'max']:
            raise ValueError(f"Mode should be either 'min' or 'max' (got {mode} instead).")

        # Create the comparison function
        if relative:
            if mode == 'min':
                self._is_better = lambda new, best: new < best - (best * tolerance)
            if mode == 'max':
                self._is_better = lambda new, best: new > best + (best * tolerance)
        else:
            if mode == 'min':
                self._is_better = lambda new, best: new < best - tolerance
            if mode == 'max':
                self._is_better = lambda new, best: new > best + tolerance

    def reset(self):
        """Reset the internal state."""
        self._best_value = self.validation_fn()
        self._num_bad_epochs = 0
        self._time_to_save = False

    def next_step(self):
        """Should be called at every iteration."""
        last_value = self.validation_fn()
        if self._is_better(last_value, self._best_value):
            self._time_to_save = True
            self._best_value = last_value
            self._num_bad_epochs = 0
        else:
            self._num_bad_epochs += 1

    def should_save(self):
        """Says if it's time to save model weights."""
        if self._time_to_save:
            self._time_to_save = False
            return True
        else:
            return False

    def should_stop(self):
        """Says if it's time to stop training."""
        return self._num_bad_epochs > self.patience
