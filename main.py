from train import Trainer
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
