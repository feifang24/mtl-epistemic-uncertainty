# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


# This script reuses some code from
# https://github.com/huggingface/transformers

import logging
import os
import pathlib
import sys
from datetime import datetime

import wandb
from sklearn.utils.class_weight import compute_sample_weight
from scipy.special import softmax

import numpy as np
import random
import tensorflow.io.gfile as gfile
import torch
import torch.nn.functional as F
import torch.optim as optim
from apex import amp
from fairseq.models.roberta import RobertaModel as FairseqRobertModel
from pytorch_pretrained_bert import BertAdam as Adam
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from transformers import (
    AlbertConfig,
    AlbertModel,
    AlbertTokenizer,
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    BertTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaModel,
    XLMRobertaTokenizer,
    XLNetConfig,
    XLNetModel,
    XLNetTokenizer,
)

from mtdnn.common.archive_maps import PRETRAINED_MODEL_ARCHIVE_MAP
from mtdnn.common.average_meter import AverageMeter
from mtdnn.common.bert_optim import Adamax, RAdam
from mtdnn.common.dropout_wrapper import DropoutWrapper
from mtdnn.common.linear_pooler import LinearPooler
from mtdnn.common.loss import LOSS_REGISTRY
from mtdnn.common.metrics import calc_metrics
from mtdnn.common.san import SANBERTNetwork, SANClassifier
from mtdnn.common.uncertainty import BatchBALD
from mtdnn.common.san_model import SanModel
from mtdnn.common.squad_utils import extract_answer, merge_answers, select_answers
from mtdnn.common.types import DataFormat, EncoderModelType, TaskType
from mtdnn.common.utils import MTDNNCommonUtils
from mtdnn.configuration_mtdnn import MTDNNConfig
from mtdnn.process_mtdnn import MTDNNDataProcess
from mtdnn.dataset_mtdnn import MTDNNCollater
from mtdnn.tasks.config import MTDNNTaskDefs
from mtdnn.tasks.utils import submit


logger = MTDNNCommonUtils.create_logger(__name__, to_disk=True)

# Supported Model Classes Map
MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetModel, XLNetTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer),
    "san": (BertConfig, SanModel, BertTokenizer),
}


class MTDNNPretrainedModel(nn.Module):
    config_class = MTDNNConfig
    pretrained_model_archive_map = PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = lambda model, config, path: None
    base_model_prefix = "mtdnn"

    def __init__(self, config):
        super(MTDNNPretrainedModel, self).__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. "
                "To create a model from a pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        # Save config in model
        self.config = config


class MTDNNModel(MTDNNPretrainedModel):
    """Instance of an MTDNN Model

    Arguments:
        MTDNNPretrainedModel {BertPretrainedModel} -- Inherited from Bert Pretrained
        config  {MTDNNConfig} -- MTDNN Configuration Object
        pretrained_model_name {str} -- Name of the pretrained model to initial checkpoint
        num_train_step  {int} -- Number of steps to take each training

    Raises:
        RuntimeError: [description]
        ImportError: [description]

    Returns:
        MTDNNModel -- An Instance of an MTDNN Model
    """

    def __init__(
        self,
        config: MTDNNConfig,
        task_defs: MTDNNTaskDefs,
        data_processor: MTDNNDataProcess,
        pretrained_model_name: str = "mtdnn-base-uncased",
        test_datasets_list: list = [],
        output_dir: str = "checkpoint",
    ):

        # Input validation
        assert (
            config.init_checkpoint in self.supported_init_checkpoints()
        ), f"Initial checkpoint must be in {self.supported_init_checkpoints()}"

        num_train_step = data_processor.get_num_all_batches()
        decoder_opts = data_processor.get_decoder_options_list()
        task_types = data_processor.get_task_types_list()
        dropout_list = data_processor.get_tasks_dropout_prob_list()
        loss_types = data_processor.get_loss_types_list()
        kd_loss_types = data_processor.get_kd_loss_types_list()
        tasks_nclass_list = data_processor.get_task_nclass_list()

        # data loaders
        multitask_train_dataloader = data_processor.get_train_dataloader()
        dev_dataloaders_list = data_processor.get_dev_dataloaders()
        test_dataloaders_list = data_processor.get_test_dataloaders()

        assert decoder_opts, "Decoder options list is required!"
        assert task_types, "Task types list is required!"
        assert dropout_list, "Task dropout list is required!"
        assert loss_types, "Loss types list is required!"
        assert kd_loss_types, "KD Loss types list is required!"
        assert tasks_nclass_list, "Tasks nclass list is required!"
        assert (
            multitask_train_dataloader
        ), "DataLoader for multiple tasks cannot be None"

        super(MTDNNModel, self).__init__(config)

        # Initialize model config and update with training options
        self.config = config
        self.update_config_with_training_opts(
            decoder_opts,
            task_types,
            dropout_list,
            loss_types,
            kd_loss_types,
            tasks_nclass_list,
        )
        wandb.init(project='mtl-uncertainty-final', entity='feifang24', config=self.config.to_dict())
        self.tasks = data_processor.tasks # {task_name: task_idx}
        self.task_defs = task_defs
        self.multitask_train_dataloader = multitask_train_dataloader
        self.dev_dataloaders_list = dev_dataloaders_list
        self.test_dataloaders_list = test_dataloaders_list
        self.test_datasets_list = self._configure_test_ds(test_datasets_list)
        self.output_dir = output_dir

        self.batch_bald = BatchBALD(num_samples=10, num_draw=500, shuffle_prop=0.0, reverse=True, reduction='mean')
        self.loss_weights = [None] * self.num_tasks

        # Create the output_dir if it's doesn't exist
        MTDNNCommonUtils.create_directory_if_not_exists(self.output_dir)

        self.pooler = None

        # Resume from model checkpoint
        if self.config.resume and self.config.model_ckpt:
            assert os.path.exists(
                self.config.model_ckpt
            ), "Model checkpoint does not exist"
            logger.info(f"loading model from {self.config.model_ckpt}")
            self = self.load(self.config.model_ckpt)
            return

        # Setup the baseline network
        # - Define the encoder based on config options
        # - Set state dictionary based on configuration setting
        # - Download pretrained model if flag is set
        # TODO - Use Model.pretrained_model() after configuration file is hosted.
        if self.config.use_pretrained_model:
            with MTDNNCommonUtils.download_path() as file_path:
                path = pathlib.Path(file_path)
                self.local_model_path = MTDNNCommonUtils.maybe_download(
                    url=self.pretrained_model_archive_map[pretrained_model_name],
                    log=logger,
                )
            self.bert_model = MTDNNCommonUtils.load_pytorch_model(self.local_model_path)
            self.state_dict = self.bert_model["state"]
        else:
            # Set the config base on encoder type set for initial checkpoint
            if config.encoder_type == EncoderModelType.BERT:
                self.bert_config = BertConfig.from_dict(self.config.to_dict())
                self.bert_model = BertModel.from_pretrained(self.config.init_checkpoint)
                self.state_dict = self.bert_model.state_dict()
                self.config.hidden_size = self.bert_config.hidden_size
            if config.encoder_type == EncoderModelType.ROBERTA:
                # Download and extract from PyTorch hub if not downloaded before
                self.bert_model = torch.hub.load(
                    "pytorch/fairseq", config.init_checkpoint
                )
                self.config.hidden_size = self.bert_model.args.encoder_embed_dim
                self.pooler = LinearPooler(self.config.hidden_size)
                new_state_dict = {}
                for key, val in self.bert_model.state_dict().items():
                    if key.startswith(
                        "model.decoder.sentence_encoder"
                    ) or key.startswith("model.classification_heads"):
                        key = f"bert.{key}"
                        new_state_dict[key] = val
                    # backward compatibility PyTorch <= 1.0.0
                    if key.startswith("classification_heads"):
                        key = f"bert.model.{key}"
                        new_state_dict[key] = val
                self.state_dict = new_state_dict

        self.updates = (
            self.state_dict["updates"]
            if self.state_dict and "updates" in self.state_dict
            else 0
        )
        self.local_updates = 0
        self.train_loss = AverageMeter()
        self.train_loss_by_task = [AverageMeter() for _ in range(len(self.tasks))]
        self.network = SANBERTNetwork(
            init_checkpoint_model=self.bert_model,
            pooler=self.pooler,
            config=self.config,
        )
        if self.state_dict:
            self.network.load_state_dict(self.state_dict, strict=False)
        self.mnetwork = (
            nn.DataParallel(self.network) if self.config.multi_gpu_on else self.network
        )
        self.total_param = sum(
            [p.nelement() for p in self.network.parameters() if p.requires_grad]
        )

        # Move network to GPU if device available and flag set
        if self.config.cuda:
            self.network.cuda(device=self.config.cuda_device)
        self.optimizer_parameters = self._get_param_groups()
        self._setup_optim(self.optimizer_parameters, self.state_dict, num_train_step)
        self.para_swapped = False
        self.optimizer.zero_grad()
        self._setup_lossmap()

    @property
    def num_tasks(self):
        return len(self.tasks)

    def _configure_test_ds(self, test_datasets_list):
        if test_datasets_list: return test_datasets_list
        result = []
        for task in self.task_defs.get_task_names():
            if task == 'mnli':
                result.append('mnli_matched')
                result.append('mnli_mismatched')
            else:
                result.append(task)
        return result


    def _get_param_groups(self):
        no_decay = ["bias", "gamma", "beta", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in self.network.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.network.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_parameters

    def _setup_optim(
        self, optimizer_parameters, state_dict: dict = None, num_train_step: int = -1
    ):

        # Setup optimizer parameters
        if self.config.optimizer == "sgd":
            self.optimizer = optim.SGD(
                optimizer_parameters,
                self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adamax":
            self.optimizer = Adamax(
                optimizer_parameters,
                self.config.learning_rate,
                warmup=self.config.warmup,
                t_total=num_train_step,
                max_grad_norm=self.config.grad_clipping,
                schedule=self.config.warmup_schedule,
                weight_decay=self.config.weight_decay,
            )

        elif self.config.optimizer == "radam":
            self.optimizer = RAdam(
                optimizer_parameters,
                self.config.learning_rate,
                warmup=self.config.warmup,
                t_total=num_train_step,
                max_grad_norm=self.config.grad_clipping,
                schedule=self.config.warmup_schedule,
                eps=self.config.adam_eps,
                weight_decay=self.config.weight_decay,
            )

            # The current radam does not support FP16.
            self.config.fp16 = False
        elif self.config.optimizer == "adam":
            self.optimizer = Adam(
                optimizer_parameters,
                lr=self.config.learning_rate,
                warmup=self.config.warmup,
                t_total=num_train_step,
                max_grad_norm=self.config.grad_clipping,
                schedule=self.config.warmup_schedule,
                weight_decay=self.config.weight_decay,
            )

        else:
            raise RuntimeError(f"Unsupported optimizer: {self.config.optimizer}")

        # Clear scheduler for certain optimizer choices
        if self.config.optimizer in ["adam", "adamax", "radam"]:
            if self.config.have_lr_scheduler:
                self.config.have_lr_scheduler = False

        if state_dict and "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])

        if self.config.fp16:
            try:
                global amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
                )
            model, optimizer = amp.initialize(
                self.network, self.optimizer, opt_level=self.config.fp16_opt_level
            )
            self.network = model
            self.optimizer = optimizer

        if self.config.have_lr_scheduler:
            if self.config.scheduler_type == "rop":
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer, mode="max", factor=self.config.lr_gamma, patience=3
                )
            elif self.config.scheduler_type == "exp":
                self.scheduler = ExponentialLR(
                    self.optimizer, gamma=self.config.lr_gamma or 0.95
                )
            else:
                milestones = [
                    int(step)
                    for step in (self.config.multi_step_lr or "10,20,30").split(",")
                ]
                self.scheduler = MultiStepLR(
                    self.optimizer, milestones=milestones, gamma=self.config.lr_gamma
                )
        else:
            self.scheduler = None

    def _setup_lossmap(self):
        self.task_loss_criterion = []
        for idx, cs in enumerate(self.config.loss_types):
            assert cs is not None, "Loss type must be defined."
            lc = LOSS_REGISTRY[cs](name=f"Loss func of task {idx}: {cs}")
            self.task_loss_criterion.append(lc)

    def _setup_kd_lossmap(self):
        loss_types = self.config.kd_loss_types
        self.kd_task_loss_criterion = []
        if self.config.mkd_opt > 0:
            for idx, cs in enumerate(loss_types):
                assert cs, "Loss type must be defined."
                lc = LOSS_REGISTRY[cs](name="Loss func of task {}: {}".format(idx, cs))
                self.kd_task_loss_criterion.append(lc)

    def _to_cuda(self, tensor):
        # Set tensor to gpu (non-blocking) if a PyTorch tensor
        if tensor is None:
            return tensor

        if isinstance(tensor, list) or isinstance(tensor, tuple):
            y = [
                e.cuda(device=self.config.cuda_device, non_blocking=True)
                for e in tensor
            ]
            for t in y:
                t.requires_grad = False
        else:
            y = tensor.cuda(device=self.config.cuda_device, non_blocking=True)
            y.requires_grad = False
        return y

    def train(self):
        if self.para_swapped:
            self.para_swapped = False

    def update(self, batch_meta, batch_data):
        self.network.train()
        target = batch_data[batch_meta["label"]]
        soft_labels = None

        task_type = batch_meta["task_type"]
        target = self._to_cuda(target) if self.config.cuda else target

        task_id = batch_meta["task_id"]
        inputs = batch_data[: batch_meta["input_len"]]
        if len(inputs) == 3:
            inputs.append(None)
            inputs.append(None)
        inputs.append(task_id)
        weight = self.loss_weights[task_id]
        if self.config.weighted_on:
            if self.config.cuda:
                weight = batch_data[batch_meta["factor"]].cuda(
                    device=self.config.cuda_device, non_blocking=True
                )
            else:
                weight = batch_data[batch_meta["factor"]]
        logits = self.mnetwork(*inputs)

        # compute loss
        loss = 0
        if self.task_loss_criterion[task_id] and (target is not None):
            loss = self.task_loss_criterion[task_id](
                logits, target, weight, ignore_index=-1
            )

        # compute kd loss
        if self.config.mkd_opt > 0 and ("soft_label" in batch_meta):
            soft_labels = batch_meta["soft_label"]
            soft_labels = (
                self._to_cuda(soft_labels) if self.config.cuda else soft_labels
            )
            kd_lc = self.kd_task_loss_criterion[task_id]
            kd_loss = (
                kd_lc(logits, soft_labels, weight, ignore_index=-1) if kd_lc else 0
            )
            loss = loss + kd_loss

        self.train_loss_by_task[task_id].update(loss.item() / (self.loss_weights[task_id] if self.loss_weights[task_id] is not None else 1.), batch_data[batch_meta["token_id"]].size(0))
        self.train_loss.update(loss.item(), batch_data[batch_meta["token_id"]].size(0))
        # scale loss
        loss = loss / (self.config.grad_accumulation_step or 1)
        if self.config.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.local_updates += 1
        if self.local_updates % self.config.grad_accumulation_step == 0:
            if self.config.global_grad_clipping > 0:
                if self.config.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(self.optimizer),
                        self.config.global_grad_clipping,
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(), self.config.global_grad_clipping
                    )
            self.updates += 1
            # reset number of the grad accumulation
            self.optimizer.step()
            self.optimizer.zero_grad()

    def eval_mode(
        self,
        data: DataLoader,
        metric_meta,
        use_cuda=True,
        with_label=True,
        label_mapper=None,
        task_type=TaskType.Classification
    ):
        eval_loss = AverageMeter()
        if use_cuda:
            self.cuda()
        predictions = []
        golds = []
        scores = []
        uncertainties = []
        ids = []
        metrics = {}
        for idx, (batch_info, batch_data) in enumerate(data):
            if idx % 100 == 0:
                logger.info(f"predicting {idx}")
            batch_info, batch_data = MTDNNCollater.patch_data(
                use_cuda, batch_info, batch_data
            )
            score, pred, gold, loss, uncertainty = self._predict_batch(batch_info, batch_data)
            predictions.extend(pred)
            golds.extend(gold)
            scores.extend(score)
            uncertainties.extend(uncertainty)
            ids.extend(batch_info["uids"])
            eval_loss.update(loss.item(), len(batch_info["uids"]))

        if task_type == TaskType.Span:
            golds = merge_answers(ids, golds)
            predictions, scores = select_answers(ids, predictions, scores)
        if with_label:
            metrics = calc_metrics(
                metric_meta, golds, predictions, scores, label_mapper
            )
        return metrics, predictions, scores, golds, ids, (eval_loss.avg, eval_loss.count), np.mean(uncertainties)

    def _predict_batch(self, batch_meta, batch_data):
        self.network.eval()
        task_id = batch_meta["task_id"]
        task_type = batch_meta["task_type"]
        inputs = batch_data[: batch_meta["input_len"]]
        if len(inputs) == 3:
            inputs.append(None)
            inputs.append(None)
        inputs.append(task_id)

        # get logits (and val loss if we have labels)
        label = batch_meta["label"]
        target = batch_data[label] if type(label) is int else torch.tensor(label)
        target = self._to_cuda(target) if self.config.cuda else target

        weight = None
        if self.config.weighted_on:
            if self.config.cuda:
                weight = batch_data[batch_meta["factor"]].cuda(
                    device=self.config.cuda_device, non_blocking=True
                )
            else:
                weight = batch_data[batch_meta["factor"]]

        score = self.mnetwork(*inputs)
        if self.config.mc_dropout_samples > 0:
            def apply_dropout(m):
                if isinstance(m, DropoutWrapper):
                    m.train()
            self.network.apply(apply_dropout)
            mc_sample_scores = torch.stack([self.mnetwork(*inputs) for _ in range(self.config.mc_dropout_samples)], -1)
            mc_sample_scores = F.softmax(mc_sample_scores, dim=1).data.cpu().numpy()
            uncertainty = self.batch_bald.get_uncertainties(mc_sample_scores)
        else:
            uncertainty = 1.0

        loss = None
        if self.task_loss_criterion[task_id] and (target is not None):
            loss = self.task_loss_criterion[task_id](
                score, target, weight, ignore_index=-1
            )

        if task_type == TaskType.Ranking:
            score = score.contiguous().view(-1, batch_meta["pairwise_size"])
            assert task_type == TaskType.Ranking
            score = F.softmax(score, dim=1)
            score = score.data.cpu()
            score = score.numpy()
            predict = np.zeros(score.shape, dtype=int)
            positive = np.argmax(score, axis=1)
            for idx, pos in enumerate(positive):
                predict[idx, pos] = 1
            predict = predict.reshape(-1).tolist()
            score = score.reshape(-1).tolist()
            return score, predict, batch_meta["true_label"], loss
        elif task_type == TaskType.SequenceLabeling:
            mask = batch_data[batch_meta["mask"]]
            score = score.contiguous()
            score = score.data.cpu()
            score = score.numpy()
            predict = np.argmax(score, axis=1).reshape(mask.size()).tolist()
            valied_lenght = mask.sum(1).tolist()
            final_predict = []
            for idx, p in enumerate(predict):
                final_predict.append(p[: valied_lenght[idx]])
            score = score.reshape(-1).tolist()
            return score, final_predict, batch_meta["label"], loss
        elif task_type == TaskType.Span:
            start, end = score
            predictions = []
            if self.config.encoder_type == EncoderModelType.BERT:
                scores, predictions = extract_answer(
                    batch_meta,
                    batch_data,
                    start,
                    end,
                    self.config.get("max_answer_len", 5),
                )
            return scores, predictions, batch_meta["answer"], loss
        else:
            if task_type == TaskType.Classification:
                score = F.softmax(score, dim=1)
            score = score.data.cpu()
            score = score.numpy()
            predict = np.argmax(score, axis=1).tolist()
            score = score.reshape(-1).tolist()
        return score, predict, batch_meta["label"], loss, uncertainty

    def _rerank_batches(self, batches, start_idx, task_id_to_weights, softmax_task_weights=False):
        def weights_to_probs(weights):
            if softmax_task_weights:
                probs = softmax(weights)
            else:
                probs = weights / np.sum(weights)
            return probs

        # reshuffle all batches; sort them by task_id
        new_batches = [list(self.multitask_train_dataloader) for _ in range(5)]
        for i in range(len(new_batches)):    
            random.shuffle(new_batches[i]) # this line somehow helps?
        new_batches = [b for batches in new_batches for b in batches] # flatten
        task_id_by_batch = [batch_meta["task_id"] for batch_meta, _ in new_batches]
        batches_by_task = [[] for _ in range(self.num_tasks)]
        for batch_idx, task_id in enumerate(task_id_by_batch):
            batches_by_task[task_id].append(batch_idx)

        task_probs = weights_to_probs(task_id_to_weights)

        # multiply weight by num batches per task
        # task_probs = weights_to_probs(task_id_to_weights * np.asarray([len(batches) for batches in batches_by_task]))  # comment out as see fit

        if self.config.uncertainty_based_weight:
            rel_loss_weights = (1. / task_id_to_weights)
            self.loss_weights = (rel_loss_weights * self.num_tasks / np.sum(rel_loss_weights)) * \
                                    (np.mean(self.dev_loss_by_task) / self.dev_loss_by_task)
            # self.loss_weights = rel_loss_weights * np.mean(task_id_to_weights)

        num_batches = len(batches[start_idx:])
        # sample num_batches many tasks w/ replacement
        task_indices_sampled = np.random.choice(self.num_tasks, num_batches, replace=True, p=task_probs)

        reranked_batches = [None] * num_batches
        counters = [0] * self.num_tasks
        for i, task_id in enumerate(task_indices_sampled):
            batch_idx = batches_by_task[task_id][counters[task_id] % len(batches_by_task[task_id])]
            counters[task_id] += 1
            reranked_batches[i] = new_batches[batch_idx]

        weights_by_task_name = {}
        for task_name, task_id in self.tasks.items():
            weights_by_task_name[f'task_weight/{task_name}'] = task_probs[task_id]

        return [None] * start_idx + reranked_batches, weights_by_task_name

    def fit(self, epochs=0):
        """ Fit model to training datasets """
        epochs = epochs or self.config.epochs
        logger.info(f"Total number of params: {self.total_param}")
        FIRST_STEP_TO_LOG = 10
        for epoch in range(1, epochs + 1):
            logger.info(f"At epoch {epoch}")
            logger.info(
                f"Amount of data to go over: {len(self.multitask_train_dataloader)}"
            )

            start = datetime.now()
            # Create batches and train
            batches = list(self.multitask_train_dataloader)
            if self.config.uncertainty_based_sampling and epoch > 1:
                batches, weights_by_task_name = self._rerank_batches(batches, start_idx=0, task_id_to_weights=self.smoothed_uncertainties_by_task)
            for idx in range(len(batches)):
                batch_meta, batch_data = batches[idx]
                batch_meta, batch_data = MTDNNCollater.patch_data(
                    self.config.cuda, batch_meta, batch_data
                )

                task_id = batch_meta["task_id"]
                self.update(batch_meta, batch_data)
                if (
                    self.local_updates == FIRST_STEP_TO_LOG
                    or (self.local_updates)
                    % (self.config.log_per_updates * self.config.grad_accumulation_step)
                    == 0
                ):

                    time_left = str(
                        (datetime.now() - start)
                        / (idx + 1)
                        * (len(self.multitask_train_dataloader) - idx - 1)
                    ).split(".")[0]
                    logger.info(
                        "Updates - [{0:6}] Training Loss - [{1:.5f}] Time Remaining - [{2}]".format(
                            self.updates, self.train_loss.avg, time_left
                        )
                    )
                    val_logs, uncertainties_by_task = self._eval(epoch, save_scores=False, eval_type='dev')
                    test_logs, _ = self._eval(epoch, save_scores=False, eval_type='test')
                    if self.local_updates == FIRST_STEP_TO_LOG:
                        weights_by_task_name = {f'task_weight/{task_name}': 1.0 for task_name in self.tasks}
                    else:
                        if self.local_updates == self.config.log_per_updates * self.config.grad_accumulation_step:
                            self.smoothed_uncertainties_by_task = uncertainties_by_task
                            self.initial_train_loss_by_task = np.asarray([loss.avg for loss in self.train_loss_by_task])
                        else:
                            alpha = self.config.smooth_uncertainties
                            self.smoothed_uncertainties_by_task = alpha * self.smoothed_uncertainties_by_task + \
                                                                    (1 - alpha) * uncertainties_by_task
                        if self.config.uncertainty_based_sampling and idx < len(batches) - 1:
                            batches, weights_by_task_name = self._rerank_batches(batches, start_idx=idx+1, task_id_to_weights=self.smoothed_uncertainties_by_task)
                        if self.config.rate_based_weight:
                            current_train_loss_by_task = np.asarray([loss.avg for loss in self.train_loss_by_task])
                            rate_of_training_by_task = current_train_loss_by_task / self.initial_train_loss_by_task
                            self.loss_weights = (rate_of_training_by_task / np.mean(rate_of_training_by_task)) * \
                                                    (np.mean(current_train_loss_by_task) / current_train_loss_by_task)
                    self._log_training({**val_logs, **test_logs, **weights_by_task_name})

                if self.config.save_per_updates_on and (
                    (self.local_updates)
                    % (
                        self.config.save_per_updates
                        * self.config.grad_accumulation_step
                    )
                    == 0
                ):
                    model_file = os.path.join(
                        self.output_dir, "model_{}_{}.pt".format(epoch, self.updates),
                    )
                    logger.info(f"Saving mt-dnn model to {model_file}")
                    self.save(model_file)

            # Eval and save checkpoint after each epoch
            logger.info('=' * 5 + f' End of EPOCH {epoch} ' + '=' * 5)
            logger.info(f'Train loss (epoch avg): {self.train_loss.avg}')
            val_logs, uncertainties_by_task = self._eval(epoch, save_scores=True, eval_type='dev')
            test_logs, _ = self._eval(epoch, save_scores=True, eval_type='test')
            self._log_training({**val_logs, **test_logs, **weights_by_task_name})

            # model_file = os.path.join(self.output_dir, "model_{}.pt".format(epoch))
            # logger.info(f"Saving mt-dnn model to {model_file}")
            # self.save(model_file)

    def _eval(self, epoch, save_scores, eval_type='dev'):
        if eval_type not in {'dev', 'test'}:
            raise ValueError("eval_type must be one of the following: 'dev' or 'test'.")
        is_dev = eval_type == 'dev'

        log_dict = {}
        loss_agg = AverageMeter()
        loss_by_task = {}
        uncertainties_by_task = {}
        for idx, dataset in enumerate(self.test_datasets_list):
            logger.info(f"Evaluating on {eval_type} ds {idx}: {dataset.upper()}")
            prefix = dataset.split("_")[0]
            results = self._predict(idx, prefix, dataset, eval_type=eval_type, saved_epoch_idx=epoch, save_scores=save_scores)

            avg_loss = results['avg_loss']
            num_samples = results['num_samples']
            loss_agg.update(avg_loss, n=num_samples)
            loss_by_task[dataset] = avg_loss
            if is_dev: logger.info(f"Task {dataset} -- {eval_type} loss: {avg_loss:.3f}")

            metrics = results['metrics']
            for key, val in metrics.items():
                if is_dev: logger.info(f"Task {dataset} -- {eval_type} {key}: {val:.3f}")
                log_dict[f'{dataset}/{eval_type}_{key}'] = val

            uncertainty = results['uncertainty']
            if is_dev: logger.info(f"Task {dataset} -- {eval_type} uncertainty: {uncertainty:.3f}")
            log_dict[f'{eval_type}_uncertainty_by_task/{dataset}'] = uncertainty
            if prefix not in uncertainties_by_task:
                uncertainties_by_task[prefix] = uncertainty
            else:
                # exploiting the fact that only mnli has two dev sets
                uncertainties_by_task[prefix] += uncertainty
                uncertainties_by_task[prefix] /= 2
        if is_dev: logger.info(f'{eval_type} loss: {loss_agg.avg}')
        log_dict[f'{eval_type}_loss'] = loss_agg.avg
        log_dict.update({f'{eval_type}_loss_by_task/{task}': loss
                    for task, loss in loss_by_task.items()})

        loss_by_task_id = [None] * self.num_tasks
        for task_name, loss in loss_by_task.items():
            loss_by_task_id[self.tasks[task_name]] = loss
        loss_by_task_id = np.asarray(loss_by_task_id)

        if is_dev:
            self.dev_loss_by_task = loss_by_task_id
        else:
            self.test_loss_by_task = loss_by_task_id
        
        # convert uncertainties_by_task from dict to list, where list[i] = weight of task_id i
        uncertainties_by_task_id = [None] * self.num_tasks
        for task_name, weight in uncertainties_by_task.items():
            task_id = self.tasks[task_name]
            uncertainties_by_task_id[task_id] = weight
        uncertainties_by_task_id = np.asarray(uncertainties_by_task_id)

        return log_dict, uncertainties_by_task_id

    def _log_training(self, val_logs):
        train_loss_by_task = {f'train_loss_by_task/{task}': self.train_loss_by_task[task_idx].avg
                                for task, task_idx in self.tasks.items()
                             }
        train_loss_agg = {'train_loss': self.train_loss.avg}
        loss_weights_by_task = {} 
        if self.config.uncertainty_based_weight or self.config.rate_based_weight:
            for task_name, task_id in self.tasks.items():
                loss_weights_by_task[f'loss_weight/{task_name}'] = self.loss_weights[task_id] if self.loss_weights[task_id] is not None else 1.
        log_dict = {'global_step': self.updates, **train_loss_by_task, **train_loss_agg, **val_logs, **loss_weights_by_task}
        wandb.log(log_dict)


    def _predict(self, eval_ds_idx, eval_ds_prefix, eval_ds_name, eval_type='dev', saved_epoch_idx=None, save_scores=True):
        if eval_type not in {'dev', 'test'}:
            raise ValueError("eval_type must be one of the following: 'dev' or 'test'.")
        is_dev = eval_type == 'dev'

        label_dict = self.task_defs.global_map.get(eval_ds_prefix, None)

        if is_dev:
            data: DataLoader = self.dev_dataloaders_list[eval_ds_idx]
        else:
            data: DataLoader = self.test_dataloaders_list[eval_ds_idx]

        if data is None:
            results = None
        else:
            with torch.no_grad():
                (
                    metrics,
                    predictions,
                    scores,
                    golds,
                    ids,
                    (eval_ds_avg_loss, eval_ds_num_samples),
                    uncertainty,
                ) = self.eval_mode(
                    data,
                    metric_meta=self.task_defs.metric_meta_map[eval_ds_prefix],
                    use_cuda=self.config.cuda,
                    with_label=True,
                    label_mapper=label_dict,
                    task_type=self.task_defs.task_type_map[eval_ds_prefix]
                )
            results = {"metrics": metrics,
                        "predictions": predictions,
                        "uids": ids,
                        "scores": scores,
                        "uncertainty": uncertainty
                        }
            if save_scores:
                score_file_prefix = f"{eval_ds_name}_{eval_type}_scores" \
                                    + (f'_{saved_epoch_idx}' if saved_epoch_idx is not None else "")
                score_file = os.path.join(self.output_dir, score_file_prefix + ".json")
                MTDNNCommonUtils.dump(score_file, results)
                if self.config.use_glue_format:
                    official_score_file = os.path.join(self.output_dir, score_file_prefix + ".tsv")
                    submit(official_score_file, results, label_dict)

            results.update({"avg_loss": eval_ds_avg_loss, "num_samples": eval_ds_num_samples})

        return results



    def predict(self, trained_model_chckpt: str = None):
        """
        Inference of model on test datasets
        """
        # Load a trained checkpoint if a valid model checkpoint
        if trained_model_chckpt and gfile.exists(trained_model_chckpt):
            logger.info(f"Running predictions using: {trained_model_chckpt}. This may take 3 minutes.")
            self.load(trained_model_chckpt)
            logger.info("Checkpoint loaded.")
        
        self.config.batch_size_eval = 128
        self.config.use_glue_format = True

        # test eval
        for idx, dataset in enumerate(self.test_datasets_list):
            prefix = dataset.split("_")[0]
            results = self._predict(idx, prefix, dataset, eval_type='test')
            if results:
                logger.info(f"[new test scores saved for {dataset}.]")
            else:
                logger.info(f"Data not found for {dataset}.")


    def extract(self, batch_meta, batch_data):
        self.network.eval()
        # 'token_id': 0; 'segment_id': 1; 'mask': 2
        inputs = batch_data[:3]
        all_encoder_layers, pooled_output = self.mnetwork.bert(*inputs)
        return all_encoder_layers, pooled_output

    def save(self, filename):
        network_state = dict(
            [(k, v.cpu()) for k, v in self.network.state_dict().items()]
        )
        params = {
            "state": network_state,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }
        torch.save(params, gfile.GFile(filename, mode='wb'))
        logger.info("model saved to {}".format(filename))

    def load(self, checkpoint):
        model_state_dict = torch.load(gfile.GFile(checkpoint, mode='rb'))
        self.network.load_state_dict(model_state_dict["state"], strict=False)
        self.optimizer.load_state_dict(model_state_dict["optimizer"])
        self.config = model_state_dict["config"]

    def cuda(self):
        self.network.cuda(device=self.config.cuda_device)

    def supported_init_checkpoints(self):
        """List of allowed check points
        """
        return [
            "bert-base-uncased",
            "bert-base-cased",
            "bert-large-uncased",
            "mtdnn-base-uncased",
            "mtdnn-large-uncased",
            "roberta.base",
            "roberta.large",
        ]

    def update_config_with_training_opts(
        self,
        decoder_opts,
        task_types,
        dropout_list,
        loss_types,
        kd_loss_types,
        tasks_nclass_list,
    ):
        # Update configurations with options obtained from preprocessing training data
        setattr(self.config, "decoder_opts", decoder_opts)
        setattr(self.config, "task_types", task_types)
        setattr(self.config, "tasks_dropout_p", dropout_list)
        setattr(self.config, "loss_types", loss_types)
        setattr(self.config, "kd_loss_types", kd_loss_types)
        setattr(self.config, "tasks_nclass_list", tasks_nclass_list)
