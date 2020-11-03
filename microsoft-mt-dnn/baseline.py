import argparse
import json
import os
import shutil
import sys
from tempfile import TemporaryDirectory

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
BATCH_SIZE = 16
MULTI_GPU_ON = False
MAX_SEQ_LEN = 128
NUM_EPOCHS = 5


def train_model(data_dir, output_dir='checkpoint', log_dir='tensorboard_logdir'):
    # Define Configuration, Tasks and Model Objects
    ROOT_DIR = TemporaryDirectory().name
    OUTPUT_DIR = os.path.join(ROOT_DIR, output_dir)
    os.makedirs(OUTPUT_DIR) if not os.path.exists(OUTPUT_DIR) else OUTPUT_DIR

    LOG_DIR = os.path.join(ROOT_DIR, log_dir)
    os.makedirs(LOG_DIR) if not os.path.exists(LOG_DIR) else LOG_DIR

    TASK_DATA_DIRS = {
        'qqp': os.path.join(data_dir, "QQP"),
        'mnli': os.path.join(data_dir, "MNLI"),
        'sst': os.path.join(data_dir, "SST-2"),
        'mrpc': os.path.join(data_dir, "MRPC")
        }

    config = MTDNNConfig(batch_size=BATCH_SIZE,
                         max_seq_len=MAX_SEQ_LEN,
                         multi_gpu_on=MULTI_GPU_ON)

    default_data_process_opts = {"header": True, "is_train": True, "multi_snli": False,}
    default_split_names = ["train", "dev", "test"]
    tasks_params = {
        "mrpc": {
                    "task_name": "mrpc",
                    "data_format": "PremiseAndOneHypothesis",
                    "encoder_type": "BERT",
                    "enable_san": True,
                    "metric_meta": ["ACC", "F1"],
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
                    "loss": "CeCriterion",
                    "kd_loss": "MseCriterion",
                    "n_class": 2,
                    "split_names": default_split_names,
                    "data_source_dir": TASK_DATA_DIRS['sst'],
                    "data_process_opts": default_data_process_opts,
                    "task_type": "Classification",
                },
        "mnli": {
            "data_format": "PremiseAndOneHypothesis",
            "encoder_type": "BERT",
            "dropout_p": 0.3,
            "enable_san": True,
            "labels": ["contradiction", "neutral", "entailment"],
            "metric_meta": ["ACC"],
            "loss": "CeCriterion",
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
    dev_dataloaders_list = data_processor.get_dev_dataloaders()
    test_dataloaders_list = data_processor.get_test_dataloaders()

    decoder_opts = data_processor.get_decoder_options_list()
    task_types = data_processor.get_task_types_list()
    dropout_list = data_processor.get_tasks_dropout_prob_list()
    loss_types = data_processor.get_loss_types_list()
    kd_loss_types = data_processor.get_kd_loss_types_list()
    tasks_nclass_list = data_processor.get_task_nclass_list()

    num_all_batches = data_processor.get_num_all_batches()

    model = MTDNNModel(
        config,
        task_defs,
        pretrained_model_name="mtdnn-base-uncased",
        num_train_step=num_all_batches,
        decoder_opts=decoder_opts,
        task_types=task_types,
        dropout_list=dropout_list,
        loss_types=loss_types,
        kd_loss_types=kd_loss_types,
        tasks_nclass_list=tasks_nclass_list,
        multitask_train_dataloader=multitask_train_dataloader,
        dev_dataloaders_list=dev_dataloaders_list,
        test_dataloaders_list=test_dataloaders_list,
        output_dir=OUTPUT_DIR,
        log_dir=LOG_DIR
    )

    model.fit(epochs=NUM_EPOCHS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MT-DNN on subset of tasks.')
    parser.add_argument('--data-dir', type=str, help='Data directory')
    parser.add_argument('--output-dir', type=str, help='Checkpoint outputs directory', default='checkpoint')
    parser.add_argument('--log-dir', type=str, help='Logging directory', default='tensorboard_logdir')
    parser.add_argument('--train', action='store_true', help='Train model')
    args = parser.parse_args()

    if args.train:
        train_model(args.data_dir, args.output_dir, args.log_dir)
