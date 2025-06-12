import torch

class HardNegativeMiner:
    def __init__(self, model, device, fraction=0.2):
        """
        model: обучаемая модель
        device: CPU / GPU
        fraction: доля сложных no-trade, которые будем усиливать
        """
        self.model = model
        self.device = device
        self.fraction = fraction

    def mine(self, X_batch, y_batch):
        """
        X_batch: Tensor формы (B, T, F)
        y_batch: Tensor формы (B, T)
        """

        B, T, F = X_batch.shape
        self.model.eval()

        with torch.no_grad():
            logits = self.model(X_batch)  # (B, T, num_pairs, 3)
            logits = logits.view(-1, 3)   # (B*T, 3)
            probs = torch.softmax(logits, dim=1)

            y_flat = y_batch.view(-1)
            mask_no_trade = (y_flat == 1)
            no_trade_probs = probs[mask_no_trade]

            max_probs = torch.max(no_trade_probs[:, [0, 2]], dim=1).values
            num_hard = int(self.fraction * len(max_probs))
            if num_hard == 0:
                return X_batch, y_batch  # ничего не меняем

            hard_indices_flat = torch.topk(max_probs, num_hard).indices
            full_flat_indices = torch.arange(y_flat.size(0), device=y_flat.device)[mask_no_trade][hard_indices_flat]

            # Преобразуем flat индексы обратно в батчевые индексы
            batch_indices = full_flat_indices // T

            unique_batches, counts = torch.unique(batch_indices, return_counts=True)

            # Собираем полные батчи, которые содержат сложные no-trade
            X_hard = X_batch[unique_batches]
            y_hard = y_batch[unique_batches]

            # Добавляем эти батчи к основным
            X_aug = torch.cat([X_batch, X_hard], dim=0)
            y_aug = torch.cat([y_batch, y_hard], dim=0)

            return X_aug, y_aug
