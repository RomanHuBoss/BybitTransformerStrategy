import math

import torch
import torch.nn as nn
from config import Config
import logging

logging.basicConfig(level=Config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Позиционное кодирование для временных рядов."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Аргументы:
            x: Тензор формы [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)]
        return x


class TransformerModel(nn.Module):
    """Модель Transformer для прогнозирования вероятностей TP/SL."""

    """Модель Transformer для прогнозирования вероятностей TP/SL."""

    def __init__(self, input_dim: int, num_sl_tp_pairs: int):
        super().__init__()
        self.num_sl_tp_pairs = num_sl_tp_pairs

        # Параметры модели
        self.input_dim = input_dim
        self.d_model = 128
        self.nhead = 8
        self.num_layers = 4
        self.dim_feedforward = 256
        self.dropout = 0.2

        # Входной слой для нормализации и проекции в d_model
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.d_model)
        )
        self.pos_encoder = PositionalEncoding(self.d_model)

        # Слой Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Общий выходной слой для всех пар SL/TP
        self.output_layer = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, num_sl_tp_pairs),
        )

        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов модели."""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if 'input_proj' in name and ('.0.' in name or '.2.' in name):
                    nn.init.xavier_uniform_(p)
                elif 'output_layer' in name and ('.0.' in name or '.2.' in name):
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Аргументы:
            x: Тензор формы [batch_size, seq_len, input_dim]

        Возвращает:
            Тензор формы [batch_size, num_sl_tp_pairs] с вероятностями для каждой пары SL/TP
        """
        # Проекция входных данных
        x = self.input_proj(x)

        # Добавление позиционного кодирования
        x = self.pos_encoder(x)

        # Обработка Transformer
        x = self.transformer(x)

        # Используем только последний элемент последовательности для прогноза
        x = x[:, -1, :]

        # Прогноз для всех пар SL/TP
        outputs = self.output_layer(x)

        return outputs

    def save(self, path: str):
        """Сохраняет модель в файл с полной информацией о конфигурации.

        Аргументы:
            path: Путь для сохранения файла модели

        Логирует:
            - Успешное сохранение модели
            - Параметры модели
        """
        try:
            # Подготовка данных для сохранения
            save_data = {
                'model_state_dict': self.state_dict(),
                'input_dim': self.input_proj[0].in_features,  # Размерность входа первого слоя
                'num_sl_tp_pairs': self.num_sl_tp_pairs,
                'model_config': {
                    'd_model': self.d_model,
                    'nhead': self.nhead,
                    'num_layers': self.num_layers,
                    'dim_feedforward': self.dim_feedforward,
                    'dropout': self.dropout
                }
            }

            # Сохранение модели
            torch.save(save_data, path)

            # Логирование успешного сохранения
            logger.info(f"Модель успешно сохранена в {path}")
            logger.debug(f"Параметры сохраненной модели: "
                         f"input_dim={save_data['input_dim']}, "
                         f"num_sl_tp_pairs={save_data['num_sl_tp_pairs']}, "
                         f"d_model={save_data['model_config']['d_model']}")

        except Exception as e:
            logger.error(f"Ошибка при сохранении модели: {str(e)}")
            logger.exception("Трассировка стека:")
            raise

    @classmethod
    def load(cls, path: str):
        """Загружает модель из файла с проверкой совместимости.

        Аргументы:
            path: Путь к файлу с сохраненной моделью

        Возвращает:
            Загруженный экземпляр TransformerModel

        Логирует:
            - Успешную загрузку
            - Параметры загруженной модели
            - Предупреждения о несоответствиях

        Исключения:
            ValueError: Если файл поврежден или структура не соответствует ожидаемой
        """
        try:
            # Загрузка данных из файла
            checkpoint = torch.load(path, map_location='cpu')
            logger.info(f"Начата загрузка модели из {path}")

            # Проверка структуры файла
            required_keys = ['model_state_dict', 'input_dim', 'num_sl_tp_pairs', 'model_config']
            if not all(key in checkpoint for key in required_keys):
                missing = [k for k in required_keys if k not in checkpoint]
                raise ValueError(f"Файл модели поврежден. Отсутствуют ключи: {missing}")

            # Создание экземпляра модели с сохраненными параметрами
            logger.debug(f"Создание модели с параметрами: "
                         f"input_dim={checkpoint['input_dim']}, "
                         f"num_sl_tp_pairs={checkpoint['num_sl_tp_pairs']}")

            model = cls(
                input_dim=checkpoint['input_dim'],
                num_sl_tp_pairs=checkpoint['num_sl_tp_pairs']
            )

            # Обработка несоответствий в state_dict
            model_state_dict = checkpoint['model_state_dict']
            current_state_dict = model.state_dict()

            # Фильтрация отсутствующих и лишних ключей
            filtered_state_dict = {}
            missing_keys = []
            unexpected_keys = []

            for k, v in model_state_dict.items():
                if k in current_state_dict:
                    if v.shape == current_state_dict[k].shape:
                        filtered_state_dict[k] = v
                    else:
                        logger.warning(f"Размерность параметра {k} не совпадает: "
                                       f"ожидалось {current_state_dict[k].shape}, "
                                       f"получено {v.shape}. Параметр будет проигнорирован.")
                else:
                    unexpected_keys.append(k)

            for k in current_state_dict:
                if k not in model_state_dict:
                    missing_keys.append(k)

            if missing_keys:
                logger.warning(f"Отсутствующие ключи в загруженной модели: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Лишние ключи в загруженной модели: {unexpected_keys}")

            # Загрузка состояния модели
            model.load_state_dict(filtered_state_dict, strict=False)
            model.eval()

            # Проверка конфигурации
            config = checkpoint['model_config']
            if (model.d_model != config['d_model'] or
                    model.nhead != config['nhead'] or
                    model.num_layers != config['num_layers']):
                logger.warning("Конфигурация модели не совпадает с сохраненной!")

            logger.info(f"Модель успешно загружена из {path}")
            logger.debug(f"Архитектура модели: "
                         f"d_model={model.d_model}, "
                         f"nhead={model.nhead}, "
                         f"num_layers={model.num_layers}, "
                         f"dim_feedforward={model.dim_feedforward}")

            return model

        except Exception as e:
            logger.error(f"Критическая ошибка при загрузке модели: {str(e)}")
            logger.exception("Трассировка стека:")
            raise ValueError(f"Не удалось загрузить модель из {path}") from e