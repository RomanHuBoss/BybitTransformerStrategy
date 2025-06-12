import numpy as np

class MarginCalibrator:
    def __init__(self, margins=None):
        """
        margins: dict вида {class: margin_shift}
        """
        self.margins = margins or {}

    def apply(self, logits):
        """
        logits: numpy array формы (N, num_classes)
        """
        adjusted = logits.copy()
        for cls, margin in self.margins.items():
            adjusted[:, cls] -= margin
        return adjusted

    def calibrate_probs(self, logits):
        adjusted_logits = self.apply(logits)
        exps = np.exp(adjusted_logits - np.max(adjusted_logits, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        return probs
