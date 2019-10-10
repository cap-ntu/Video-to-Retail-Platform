import os
import sys
import time
from multiprocessing import Pool

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(this_dir, '..'))
sys.path.insert(0, os.path.join(this_dir, '../hysia'))
sys.path.insert(0, os.path.join(this_dir, '../third'))

from hysia.utils.logger import Logger
from model_server.audio_model_server import main as audio_runner
from model_server.feature_extract_model_server import main as feature_extract_runner
from model_server.product_search_model_server import main as product_search_runner
from model_server.scene_search_model_server import main as scene_search_runner
from model_server.visual_model_server import main as visual_runner

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Time constant
_ONE_DAY_IN_SECONDS = 24 * 60 * 60

model_runners = [
    audio_runner,
    feature_extract_runner,
    product_search_runner,
    scene_search_runner,
    visual_runner,
]

if __name__ == "__main__":

    logger = Logger(
        name="model_server",
        severity_levels={
            "StreamHandler": "INFO"
        }
    )
    logger.info("Starting model servers")

    with Pool(processes=len(model_runners)) as pool:
        for runner in model_runners:
            pool.apply_async(runner)

        while True:
            try:
                time.sleep(_ONE_DAY_IN_SECONDS)
            except KeyboardInterrupt:
                logger.info("Shutting down model servers")
                pool.terminate()
                exit(0)
