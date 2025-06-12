import torch

class HardNegativeMiner:
    def __init__(self, model, device, fraction=0.2):
        """
        model: текущая модель
        fraction: доля сложных отрицательных примеров, которые будем усиливать
        """
        self.model = model
        self.device = device
        self.fraction = fraction

    def mine(self, X_batch, y_batch):
        """
        X_batch: Tensor формы (B, T, F)
        y_batch: Tensor формы (B, T)
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_batch)  # (B, T, num_pairs, 3)
            logits = logits.view(-1, 3)   # (B*T, 3)
            probs = torch.softmax(logits, dim=1)

            y_flat = y_batch.view(-1)
            mask_no_trade = (y_flat == 1)
            no_trade_probs = probs[mask_no_trade]

            # ищем hard negatives — те, у кого макс softmax близок к SHORT или LONG
            max_probs = torch.max(no_trade_probs[:, [0, 2]], dim=1).values
            num_hard = int(self.fraction * len(max_probs))
            if num_hard == 0:
                return X_batch, y_batch  # ничего не меняем

            hard_indices = torch.topk(max_probs, num_hard).indices
            full_indices = torch.arange(y_flat.size(0), device=y_flat.device)[mask_no_trade][hard_indices]

            # дублируем сложные примеры в батче
            X_hard = X_batch.view(-1, *X_batch.shape[2:])[full_indices]
            y_hard = y_flat[full_indices]

            # соединяем исходный батч и хард-примеры
            X_aug = torch.cat([X_batch.view(-1, *X_batch.shape[2:]), X_hard], dim=0)
            y_aug = torch.cat([y_flat, y_hard], dim=0)

            T = y_batch.shape[1]
            F = X_batch.shape[2]
            new_batch_size = X_aug.shape[0] // T
            return X_aug.view(new_batch_size, T, F), y_aug.view(new_batch_size, T)
