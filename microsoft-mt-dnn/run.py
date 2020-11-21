import argparse
import json
import os
import shutil
import sys
from datetime import datetime

import pandas as pd
import torch

from mtdnn.common.types import EncoderModelType
from mtdnn.configuration_mtdnn import MTDNNConfig
from mtdnn.data_builder_mtdnn import MTDNNDataBuilder
from mtdnn.modeling_mtdnn import MTDNNModel
from mtdnn.process_mtdnn import MTDNNDataProcess
from mtdnn.tasks.config import MTDNNTaskDefs
from mtdnn.tokenizer_mtdnn import MTDNNTokenizer


# Training parameters
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 128
MULTI_GPU_ON = False
MAX_SEQ_LEN = 128

def train_model(data_dir, uncertainty_based_sampling=False, mc_dropout_samples=100, uncertainty_based_weight=False, rate_based_weight=False, focal_loss=False, bilstm=False, debug=False):
    # Define Configuration, Tasks and Model Objects
    ROOT_DIR = 'gs://cs330'
    MODEL_ID = datetime.now().strftime('%m%d%H%M')
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'checkpoint', MODEL_ID)
    NUM_EPOCHS = 2 if debug else 4
    LOG_PER_UPDATES = 4 if debug else 200


    TASK_DATA_DIRS = {
        'sst': os.path.join(data_dir, "SST-2"),
        'mrpc': os.path.join(data_dir, "MRPC"),
        'rte': os.path.join(data_dir, "RTE")
        }

    config = MTDNNConfig(batch_size=BATCH_SIZE,
                         batch_size_eval=EVAL_BATCH_SIZE,
                         max_seq_len=MAX_SEQ_LEN,
                         multi_gpu_on=MULTI_GPU_ON,
                         log_per_updates=LOG_PER_UPDATES,
                         uncertainty_based_sampling=uncertainty_based_sampling,
                         mc_dropout_samples=mc_dropout_samples,
                         uncertainty_based_weight=uncertainty_based_weight,
                         rate_based_weight=rate_based_weight,
                         focal_loss=focal_loss,
                         bilstm=bilstm
                        )

    default_data_process_opts = {"header": True, "is_train": True, "multi_snli": False,}
    default_split_names = ["train", "dev", "test"]
    tasks_params = {
        "mrpc": {
                    "task_name": "mrpc",
                    "data_format": "PremiseAndOneHypothesis",
                    "encoder_type": "BERT",
                    "enable_san": True,
                    "metric_meta": ["ACC", "F1", "AUC"],
                    "loss": "CeCriterion" if not focal_loss else "FocalLossCriterion",
                    "metric_meta": ["ACC", "F1", "AUC"],
                    "loss": "CeCriterion",
                    "kd_loss": "MseCriterion",
                    "n_class": 2,
                    "split_names": default_split_names,
                    "data_source_dir": TASK_DATA_DIRS['mrpc'],
                    "data_process_opts": default_data_process_opts,
                    "task_type": "Classification",
        },
        "sst": {
                    "task_name": "sst",
                    "data_format": "PremiseOnly",
                    "encoder_type": "BERT",
                    "enable_san": False,
                    "metric_meta": ["ACC"],
                    "loss": "CeCriterion" if not focal_loss else "FocalLossCriterion",
                    "metric_meta": ["ACC", "F1", "AUC"],
                    "kd_loss": "MseCriterion",
                    "n_class": 2,
                    "split_names": default_split_names,
                    "data_source_dir": TASK_DATA_DIRS['sst'],
                    "data_process_opts": default_data_process_opts,
                    "task_type": "Classification",
                },
        "rte": {
            "task_name": "rte",
            "data_format": "PremiseAndOneHypothesis",
            "encoder_type": "BERT",
            "enable_san": True,
            "labels": ["contradiction", "neutral", "entailment"],
            "metric_meta": ["ACC"],
            "loss": "CeCriterion" if not focal_loss else "FocalLossCriterion",
            "kd_loss": "MseCriterion",
            "n_class": 3,
            "split_names": [
                "train",
                "dev_matched",
                "dev_mismatched",
                "test_matched",
                "test_mismatched",
            ],
            "data_source_dir": TASK_DATA_DIRS['mnli'],
            "data_process_opts": {"header": True, "is_train": True, "multi_snli": False,},
            "task_type": "Classification",
        },
        "qqp": {
            "task_name": "qqp",
            "data_format": "PremiseAndOneHypothesis",
            "encoder_type": "BERT",
            "enable_san": True,
            "metric_meta": ["ACC", "F1"],
            "loss": "CeCriterion" if not focal_loss else "FocalLossCriterion",
            "metric_meta": ["ACC", "F1", "AUC"],
            "kd_loss": "MseCriterion",
            "n_class": 2,
            "split_names": default_split_names,
            "data_source_dir": TASK_DATA_DIRS['rte'],
            "data_process_opts": default_data_process_opts,
            "task_type": "Classification",
        }
    }

    # Define the tasks
    task_defs = MTDNNTaskDefs(tasks_params)

    ## Load and build data
    tokenizer = MTDNNTokenizer(do_lower_case=True)
    data_builder = MTDNNDataBuilder(
        tokenizer=tokenizer,
        task_defs=task_defs,
        data_dir='.', #DATA_SOURCE_DIR,
        canonical_data_suffix="canonical_data",
        dump_rows=True,
    )

    ## Build data to MTDNN Format
    ## Iterable of each specific task and processed data
    vectorized_data = data_builder.vectorize()

    # Make the Data Preprocess step and update the config with training data updates
    data_processor = MTDNNDataProcess(
        config=config, task_defs=task_defs, vectorized_data=vectorized_data
    )

    multitask_train_dataloader = data_processor.get_train_dataloader()

    model = MTDNNModel(
        config,
        task_defs,
        data_processor=data_processor,
        pretrained_model_name="bert-base-uncased",
        output_dir=OUTPUT_DIR,
    )

    model.fit(epochs=NUM_EPOCHS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MT-DNN on subset of tasks.')
    parser.add_argument('--data-dir', type=str, help='Data directory')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--uncertainty-based-sampling', action='store_true', help='Use uncertainty based batch sampling')
    parser.add_argument('--mc-dropout-samples', default=100, type=int, help='Number of MC Dropout sampling iterations.')
    parser.add_argument('--focal-loss', action='store_true', help='Use focal loss.')
    parser.add_argument('--uncertainty-based-weight', action='store_true', help='Use uncertainty based weight in loss weighting')
    parser.add_argument('--rate-based-weight', action='store_true', help='Use training rate-based weight in loss weighting')
    parser.add_argument('--bilstm', action='store_true', help='Use Bi-LSTM encoder.')
    args = parser.parse_args()

    if args.train:
        train_model(args.data_dir,
                    uncertainty_based_sampling=args.uncertainty_based_sampling,
                    mc_dropout_samples=args.mc_dropout_samples,
                    uncertainty_based_weight=args.uncertainty_based_weight,
                    rate_based_weight=args.rate_based_weight,
                    focal_loss=args.focal_loss,
                    bilstm=args.bilstm
                    debug=args.debug)
                    )
