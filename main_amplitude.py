# main_amplitude.py

from train_amplitude import AmplitudeTrainer
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    trainer = AmplitudeTrainer()
    trainer.train()
