"""
Very basic script that ties dataset creation and model running together, for use on a compute cluster!
"""

from pre_processing_SQUAD import start_generation
from src import build_model_and_train

start_generation()
print("Starting Model")
build_model_and_train()
