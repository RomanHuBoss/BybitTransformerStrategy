import numpy as np
from sklearn.metrics import f1_score
import logging

class ThresholdTuner:
    def __init__(self, thresholds=None):
        """
        thresholds: dict вида {class: threshold}
        """
        self.thresholds = thresholds

    def search_optimal_thresholds(self, probs, targets, step=0.01):
        """
        Автоматический подбор порогов для классов SHORT (0) и LONG (2).
        probs — откалиброванные вероятности (softmax после temperature scaling).
        targets — истинные метки классов.
        """
        best_thresholds = {}

        for cls in [0, 2]:  # только SHORT и LONG тюним
            best_f1 = -1
            best_th = 0.5
            for th in np.arange(0.1, 0.9, step):
                preds = self._apply_thresholds(probs, custom={cls: th})
                f1 = f1_score(targets, preds, average='macro')
                if f1 > best_f1:
                    best_f1 = f1
                    best_th = th
            best_thresholds[cls] = round(best_th, 3)
            logging.info(f"🔧 Threshold search: class {cls} → best_th={best_th:.3f} (F1={best_f1:.4f})")

        self.thresholds = best_thresholds
        return best_thresholds

    def apply_thresholds(self, probs):
        """
        Применение сохранённых порогов
        """
        if self.thresholds is None:
            raise ValueError("Thresholds not set. Call search_optimal_thresholds first or load from meta.")
        return self._apply_thresholds(probs, self.thresholds)

    def _apply_thresholds(self, probs, custom):
        """
        Внутреннее применение порогов.
        custom — dict вида {class: threshold}
        """
        preds = np.argmax(probs, axis=1)
        for cls, th in custom.items():
            mask = probs[:, cls] < th
            preds[mask & (preds == cls)] = 1  # переводим в NO-TRADE
        return preds

    def to_dict(self):
        return self.thresholds

    @classmethod
    def from_dict(cls, data):
        if data is None:
            return cls()
        data_int = {int(k): v for k, v in data.items()}
        return cls(thresholds=data_int)
