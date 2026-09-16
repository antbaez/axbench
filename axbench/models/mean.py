from .model import Model
import torch, transformers, datasets
from tqdm.auto import tqdm
import os
import pandas as pd
from pyvene import (
    IntervenableConfig,
    IntervenableModel
)
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union, List, Any
from torch.utils.data import DataLoader
from .interventions import (
    AdditionIntervention,
    SubspaceIntervention,
    JumpReLUSAECollectIntervention,
    PositionwiseAdditionIntervention
)
from ..utils.model_utils import (
    set_decoder_norm_to_unit_norm, 
    remove_gradient_parallel_to_decoder_directions,
    gather_residual_activations, 
    get_lr
)
from ..utils.model_utils import calculate_l1_losses
from transformers import get_scheduler
import sklearn.decomposition
import numpy as np

from .probe import DataCollator, make_data_module

import logging
import random
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, low_rank_dimension):
        super(LogisticRegressionModel, self).__init__()
        # Linear layer: input_dim -> 1 output (since binary classification)
        self.proj = torch.nn.Linear(input_dim, low_rank_dimension)
        with torch.no_grad():
            self.proj.bias.fill_(0)

    def forward(self, x):
        return self.proj(x)


class MeanEmbedding(Model):
    def __str__(self):
        return 'MeanEmbedding'

    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "train")
        intervention_type = kwargs.get("intervention_type", "addition")
        if mode == "steering":
            if intervention_type == "addition":
                ax = AdditionIntervention(
                embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            elif intervention_type == "clamping":
                ax = SubspaceIntervention(
                    embed_dim=self.model.config.hidden_size, 
                    low_rank_dimension=kwargs.get("low_rank_dimension", 1),
                )
            self.ax = ax
            self.ax.train()
            ax_config = IntervenableConfig(representations=[{
                "layer": l,
                "component": f"model.layers[{l}].output",
                "low_rank_dimension": kwargs.get("low_rank_dimension", 1),
                "intervention": self.ax} for l in [self.layer]])
            ax_model = IntervenableModel(ax_config, self.model)
            ax_model.set_device(self.device)
            self.ax_model = ax_model
        else:
            ax = LogisticRegressionModel(
                self.model.config.hidden_size, kwargs.get("low_rank_dimension", 1))
            ax.to(self.device)
            self.ax = ax
    
    def make_dataloader(self, examples, **kwargs):
        data_module = make_data_module(self.tokenizer, self.model, examples)
        train_dataloader = DataLoader(
            data_module["train_dataset"], shuffle=True, batch_size=self.training_args.batch_size, 
            collate_fn=data_module["data_collator"])
        return train_dataloader

    def train(self, examples, **kwargs):
        torch.cuda.empty_cache()
        # set the decoder weights to be the mean of the embeddings
        W_U = self.model.lm_head.weight.mean(dim=0).detach().clone().unsqueeze(0)
        self.ax.proj.weight.data = W_U.data
        set_decoder_norm_to_unit_norm(self.ax)
        logger.warning("Training finished.")


class MeanActivation(MeanEmbedding):
    """take the mean of all activations"""
    def __str__(self):
        return 'MeanActivation'

    @torch.no_grad()
    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples)
        torch.cuda.empty_cache()
        self.ax.eval()
        # Main training loop.
        all_activations = []
        num_training_steps = self.training_args.n_epochs * len(train_dataloader)
        for epoch in range(self.training_args.n_epochs):
            for batch in train_dataloader:
                # prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                activations = gather_residual_activations(
                    self.model, self.layer, 
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
                ).detach()
                nonbos_mask = inputs["attention_mask"][:,kwargs["prefix_length"]:]
                activations = activations[:,kwargs["prefix_length"]:][nonbos_mask.bool()]
                all_activations.append(activations)
        all_activations = torch.cat(all_activations, dim=0)
        mean_activation = all_activations.mean(dim=0)
        self.ax.proj.weight.data = mean_activation.unsqueeze(0)
        set_decoder_norm_to_unit_norm(self.ax)
        logger.warning("Training finished.")


class DiffMean(MeanActivation):
    """
    difference in means of positive and negative classes
    - https://arxiv.org/abs/2310.06824
    - https://blog.eleuther.ai/diff-in-means/
    """
    
    def __str__(self):
        return 'DiffMean'

    @torch.no_grad()
    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples)
        torch.cuda.empty_cache()
        self.ax.eval()
        self.ax.to(self.device)
        # Main training loop.
        positive_activations = []
        negative_activations = []
        for _ in range(self.training_args.n_epochs):
            for batch in train_dataloader:
                # prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                activations = gather_residual_activations(
                    self.model, self.layer, 
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
                ).detach()
                nonbos_mask = inputs["attention_mask"][:,kwargs["prefix_length"]:]
                sliced_input_ids = inputs["input_ids"][:, kwargs["prefix_length"]:]
                for i in range(min(3, sliced_input_ids.shape[0])):
                    kept_ids = sliced_input_ids[i][nonbos_mask[i].bool()]
                    print(f"[DiffMean] example {i}: {self.tokenizer.decode(kept_ids)}")
                activations = activations[:,kwargs["prefix_length"]:][nonbos_mask.bool()]
                labels = inputs["labels"].unsqueeze(1).repeat(
                    1, inputs["input_ids"].shape[1] - kwargs["prefix_length"])
                positive_activations.append(activations[labels[nonbos_mask.bool()] == 1])
                negative_activations.append(activations[labels[nonbos_mask.bool()] != 1])

        mean_positive_activation = torch.cat(positive_activations, dim=0).mean(dim=0)
        mean_negative_activation = torch.cat(negative_activations, dim=0).mean(dim=0)
        self.ax.proj.weight.data = mean_positive_activation.unsqueeze(0) - mean_negative_activation.unsqueeze(0)
        set_decoder_norm_to_unit_norm(self.ax)
        logger.warning("Training finished.")


class PCA(MeanActivation):
    
    def __str__(self):
        return 'PCA'

    @torch.no_grad()
    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples)
        torch.cuda.empty_cache()
        self.ax.eval()
        self.ax.to(self.device)
        # Main training loop.
        all_activations = []
        
        for _ in range(self.training_args.n_epochs):
            for batch in train_dataloader:
                # prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                activations = gather_residual_activations(
                    self.model, self.layer, 
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
                ).detach()
                nonbos_mask = inputs["attention_mask"][:,kwargs["prefix_length"]:]
                activations = activations[:,kwargs["prefix_length"]:][nonbos_mask.bool()]
                labels = inputs["labels"].unsqueeze(1).repeat(1, inputs["input_ids"].shape[1] - kwargs["prefix_length"])
                label_mask = labels[nonbos_mask.bool()] == 1 # only positive examples
                all_activations.append(activations[label_mask].detach().cpu().float().numpy())

        all_activations = np.concatenate(all_activations)
        pca = sklearn.decomposition.PCA(n_components=2)
        pca.fit(all_activations)
        variance = pca.explained_variance_ratio_[0]
        logger.warning(f"PCA explains {variance:.5%} of the variance")
        first_principal_component = torch.tensor(pca.components_[0])
        self.ax.proj.weight.data = first_principal_component.unsqueeze(0)
        set_decoder_norm_to_unit_norm(self.ax)
        logger.warning("Training finished.")


class LAT(MeanActivation):
    """
    LAT is just PCA over normed differences of random pairs of activations
    - https://arxiv.org/abs/2310.01405
    """
    
    def __str__(self):
        return 'LAT'

    @torch.no_grad()
    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples)
        torch.cuda.empty_cache()
        self.ax.eval()
        self.ax.to(self.device)
        # Main training loop.
        all_activations = []
        
        for _ in range(self.training_args.n_epochs):
            for batch in train_dataloader:
                # prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                activations = gather_residual_activations(
                    self.model, self.layer, 
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
                ).detach()
                nonbos_mask = inputs["attention_mask"][:,kwargs["prefix_length"]:]
                activations = activations[:,kwargs["prefix_length"]:][nonbos_mask.bool()]
                labels = inputs["labels"].unsqueeze(1).repeat(1, inputs["input_ids"].shape[1] - kwargs["prefix_length"])
                label_mask = labels[nonbos_mask.bool()] == 1 # only positive examples
                all_activations.append(activations[label_mask].detach().cpu().float().numpy())

        # shuffle and take diffs of random pairs
        all_activations = np.concatenate(all_activations)
        logger.warning(f"Shuffling {all_activations.shape[0]} activations")
        np.random.shuffle(all_activations)
        length = all_activations.shape[0] // 2
        all_activations = all_activations[:length] - all_activations[length:length * 2]
        logger.warning(f"Shuffled and diff'd:  {all_activations.shape[0]} ")
        logger.warning(f"Potential NaNs: {np.isnan(all_activations).sum()}")
        logger.warning(f"Potential Infs: {np.isinf(all_activations).sum()}")
        logger.warning(f"Range: {all_activations.min()} to {all_activations.max()}")

        # normalize the diffs, avoiding division by zero
        norms = np.linalg.norm(all_activations, axis=1, keepdims=True)
        all_activations = np.where(norms == 0, 0, all_activations / norms)

        # fit PCA on the diffs
        pca = sklearn.decomposition.PCA(n_components=2)
        pca.fit(all_activations)
        variance = pca.explained_variance_ratio_[0]
        logger.warning(f"LAT explains {variance:.5%} of the variance")
        first_principal_component = torch.tensor(pca.components_[0])
        self.ax.proj.weight.data = first_principal_component.unsqueeze(0)
        set_decoder_norm_to_unit_norm(self.ax)
        logger.warning("Training finished.")


class GemmaScopeSAEDiffMean(MeanActivation):
    """
    SAE mean difference
    """
    def __str__(self):
        return 'GemmaScopeSAEDiffMean'

    def make_model(self, **kwargs):
        if kwargs.get("mode", "latent") == "train":
            # load the entire SAE
            self.sae_params = kwargs.get("sae_params", None)
            self.metadata_path = kwargs.get("metadata_path", None)
            self.sae_width = self.sae_params['W_dec'].shape[0]
            self.sae = JumpReLUSAECollectIntervention(
                embed_dim=self.model.config.hidden_size, 
                low_rank_dimension=self.sae_width,
            )
            sae_pt_params = {k: torch.from_numpy(v) for k, v in self.sae_params.items()}
            self.sae.load_state_dict(sae_pt_params, strict=False)
            self.sae.eval()
            self.sae.to(self.device)
        super().make_model(**kwargs)
    
    @torch.inference_mode()
    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples)
        torch.cuda.empty_cache()
        prefix_length = kwargs.get("prefix_length", 1)

        sum_positive_acts = torch.zeros(self.sae_width).to(self.device)
        sum_negative_acts = torch.zeros(self.sae_width).to(self.device)
        positive_count = 0
        negative_count = 0
        num_training_steps = self.training_args.n_epochs * len(train_dataloader)
        rank = torch.distributed.get_rank()
        progress_bar, curr_step = tqdm(range(num_training_steps), position=rank, leave=True), 0
        
        for epoch in range(self.training_args.n_epochs):
            for step, batch in enumerate(train_dataloader):
                # prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                inputs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                }

                # get SAE latents
                act_in = gather_residual_activations(
                    self.model, self.layer, inputs).to(dtype=torch.float32)
                ax_acts_batch = self.sae(act_in[:, prefix_length:]) # no bos token
                seq_lens = inputs["attention_mask"].sum(dim=1) - prefix_length # no bos token

                # add avg latents for each sequence
                for i in range(len(batch)):
                    acts = ax_acts_batch[i, :seq_lens[i]]
                    label = batch["labels"][i]
                    avg_acts = acts.mean(dim=0)
                    if label == 1:
                        sum_positive_acts += avg_acts
                        positive_count += 1
                    else:
                        sum_negative_acts += avg_acts
                        negative_count += 1
                
                del ax_acts_batch
                del act_in
                torch.cuda.empty_cache()
                progress_bar.update(1)
        progress_bar.close()

        # get latent activations
        mean_positive_activation = sum_positive_acts / positive_count
        mean_negative_activation = sum_negative_acts / negative_count
        mean_diff = (mean_positive_activation - mean_negative_activation) @ self.sae.W_dec
        self.ax.proj.weight.data = mean_diff.unsqueeze(0)
        self.ax.proj.bias.data = torch.zeros(1)
        set_decoder_norm_to_unit_norm(self.ax)

        # print top 10 features
        top = (mean_positive_activation - mean_negative_activation).topk(10)
        for i in range(10):
            logger.warning(f"Feature {top.indices[i].item()}: {top.values[i].item()}")
        logger.warning("Training finished.")


@dataclass
class LeftPadDataCollator(object):
    """Left-padding counterpart of probe.DataCollator.

    Pads every sequence to one fixed width so that end-aligned indexing is a constant
    column for every row: the last real token is always at column -1, with no per-row
    gather. Named distinctly from probe.DataCollator because axbench/__init__.py star-
    imports probe before mean, so a shared name would shadow it in the axbench namespace.
    """

    tokenizer: transformers.AutoTokenizer
    data_collator: transformers.DataCollator
    max_seq_len: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        for inst in instances:
            input_ids = inst["input_ids"]
            pad_len = self.max_seq_len - len(input_ids)
            paddings = torch.tensor(
                [self.tokenizer.pad_token_id for _ in range(pad_len)])
            inst["input_ids"] = torch.cat((paddings, input_ids)).int()
            # Build the mask from the known real length rather than by comparing against
            # pad_token_id (cf. probe.DataCollator), so a real token that happens to equal
            # pad_token_id is not silently treated as padding.
            attention_mask = torch.zeros(self.max_seq_len, dtype=torch.int)
            attention_mask[pad_len:] = 1
            inst["attention_mask"] = attention_mask
            inst["labels"] = inst["labels"].int()
        return self.data_collator(instances)


def make_left_padded_data_module(
    tokenizer: transformers.PreTrainedTokenizer, model, df, max_seq_length=None
):
    """Tokenize once, then left-pad every row to a single global width.

    The width is the longest example in `df` (i.e. per training call, which AxBench makes
    per concept), optionally capped by `max_seq_length`. Truncation keeps the *end* of the
    sequence, since every variant built on this indexes backward from the last token.
    """
    all_input_ids, all_labels = [], []
    for _, row in df.iterrows():
        input_ids = tokenizer(
            row["input"], max_length=1024, truncation=True, return_tensors="pt")["input_ids"][0]
        all_input_ids.append(input_ids)
        all_labels.append(row["labels"])

    observed_max = max(len(ids) for ids in all_input_ids)
    max_seq_len = min(observed_max, max_seq_length) if max_seq_length else observed_max
    if observed_max > max_seq_len:
        logger.warning(
            f"Truncating {observed_max} -> {max_seq_len} tokens from the left "
            f"(keeping the end of each sequence).")
        all_input_ids = [ids[-max_seq_len:] for ids in all_input_ids]

    train_dataset = datasets.Dataset.from_dict({
        "input_ids": all_input_ids,
        "labels": all_labels,
    })
    train_dataset.set_format(type='torch', columns=['input_ids', 'labels'])

    data_collator_fn = transformers.DefaultDataCollator(return_tensors="pt")
    data_collator = LeftPadDataCollator(
        tokenizer=tokenizer, data_collator=data_collator_fn, max_seq_len=max_seq_len)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


class MeanTokenDiffMean(MeanActivation):
    """DiffMean over every real token, computed on left-padded batches.

    Should be numerically equivalent to DiffMean: RoPE attention depends only on relative
    position, so shifting a sequence rightward changes nothing, and padded columns are
    attended with weight zero. Exists as the explicitly-named counterpart to
    LastTokenDiffMean and as an equivalence check on the left-padding machinery.
    """

    def __str__(self):
        return 'MeanTokenDiffMean'

    def make_dataloader(self, examples, **kwargs):
        data_module = make_left_padded_data_module(
            self.tokenizer, self.model, examples,
            max_seq_length=getattr(self.training_args, "max_seq_length", None))
        g = torch.Generator()
        g.manual_seed(self.seed)
        return DataLoader(
            data_module["train_dataset"], shuffle=True,
            batch_size=self.training_args.batch_size,
            collate_fn=data_module["data_collator"], generator=g)

    @staticmethod
    def _real_token_mask(attention_mask, prefix_length):
        """True for real, non-prefix tokens in a left-padded batch.

        Under left padding each row's real content starts at a different column, so the
        prefix cannot be dropped with a fixed slice the way DiffMean does it.
        """
        seq_len = attention_mask.shape[1]
        cols = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0)
        start = (seq_len - attention_mask.sum(dim=1)).unsqueeze(1) + prefix_length
        return (cols >= start) & attention_mask.bool()

    @torch.no_grad()
    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples, **kwargs)
        torch.cuda.empty_cache()
        self.ax.eval()
        self.ax.to(self.device)
        prefix_length = kwargs["prefix_length"]

        positive_activations, negative_activations = [], []
        for _ in range(self.training_args.n_epochs):
            for batch in train_dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                activations = gather_residual_activations(
                    self.model, self.layer,
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
                ).detach()
                real_mask = self._real_token_mask(inputs["attention_mask"], prefix_length)
                for i in range(min(3, inputs["input_ids"].shape[0])):
                    kept_ids = inputs["input_ids"][i][real_mask[i]]
                    print(f"[MeanTokenDiffMean] example {i}: {self.tokenizer.decode(kept_ids)}")
                acts = activations[real_mask]
                labels = inputs["labels"].unsqueeze(1).repeat(
                    1, activations.shape[1])[real_mask]
                positive_activations.append(acts[labels == 1])
                negative_activations.append(acts[labels != 1])

        mean_positive_activation = torch.cat(positive_activations, dim=0).mean(dim=0)
        mean_negative_activation = torch.cat(negative_activations, dim=0).mean(dim=0)
        self.ax.proj.weight.data = \
            mean_positive_activation.unsqueeze(0) - mean_negative_activation.unsqueeze(0)
        set_decoder_norm_to_unit_norm(self.ax)


class LastTokenDiffMean(MeanTokenDiffMean):
    """DiffMean built from only the final real token of each sequence.

    Left padding puts that token at column -1 for every row, so this is a plain slice.
    Produces a single direction, so it steers through the stock AdditionIntervention and
    is applied at *every* position during inference -- deliberately unlike
    DiffMeanPositional(num_positions=1), which only steers the final prompt token.
    """

    def __str__(self):
        return 'LastTokenDiffMean'

    @torch.no_grad()
    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples, **kwargs)
        torch.cuda.empty_cache()
        self.ax.eval()
        self.ax.to(self.device)
        prefix_length = kwargs["prefix_length"]

        positive_activations, negative_activations = [], []
        for _ in range(self.training_args.n_epochs):
            for batch in train_dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                activations = gather_residual_activations(
                    self.model, self.layer,
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
                ).detach()
                # left padding: final real token is always the last column
                acts = activations[:, -1, :]
                labels = inputs["labels"]
                # skip rows that are nothing but prefix
                valid = inputs["attention_mask"].sum(dim=1) > prefix_length
                positive_activations.append(acts[valid & (labels == 1)])
                negative_activations.append(acts[valid & (labels != 1)])

        mean_positive_activation = torch.cat(positive_activations, dim=0).mean(dim=0)
        mean_negative_activation = torch.cat(negative_activations, dim=0).mean(dim=0)
        self.ax.proj.weight.data = \
            mean_positive_activation.unsqueeze(0) - mean_negative_activation.unsqueeze(0)
        set_decoder_norm_to_unit_norm(self.ax)


class DiffMeanPositional(MeanTokenDiffMean):
    """One DiffMean direction per end-aligned token position.

    Training indexes backward from the last real token of the (instruction + response)
    sequence. At inference the vectors are applied end-aligned to the *prompt* -- v_0 on
    the final prompt token, v_1 on the one before it -- and generation is left unsteered.
    Note the asymmetry: "the end" is end-of-response in training but end-of-prompt at test.

    Weights are [n_concepts, num_positions, hidden], so save/load are overridden. A
    collapsed mean-across-positions direction is also stored in self.ax so that latent
    mode (predict_latent / pre_compute_mean_activations / get_logits) works unchanged --
    that path is required, since steering reads {ClassName}_max_act from the latent files
    and silently falls back to 1.0 if it is missing.
    """

    def __str__(self):
        return 'DiffMeanPositional'

    def _resolve_num_positions(self, **kwargs):
        num_positions = kwargs.get("num_positions") or \
            getattr(self.training_args, "num_positions", None)
        if not num_positions:
            raise ValueError(
                "DiffMeanPositional requires num_positions. Set it under this model in the "
                "train.models section of your config.")
        return int(num_positions)

    def make_model(self, **kwargs):
        if kwargs.get("mode") != "steering":
            # train / latent: a single collapsed direction, handled by the parent
            return super().make_model(**kwargs)

        ax = PositionwiseAdditionIntervention(
            embed_dim=self.model.config.hidden_size,
            low_rank_dimension=kwargs.get("low_rank_dimension", 1),
            num_positions=self._resolve_num_positions(**kwargs),
        )
        self.ax = ax
        self.ax.train()
        ax_config = IntervenableConfig(representations=[{
            "layer": l,
            "component": f"model.layers[{l}].output",
            "low_rank_dimension": kwargs.get("low_rank_dimension", 1),
            "intervention": self.ax} for l in [self.layer]])
        ax_model = IntervenableModel(ax_config, self.model)
        ax_model.set_device(self.device)
        self.ax_model = ax_model

    @torch.no_grad()
    def train(self, examples, **kwargs):
        num_positions = self._resolve_num_positions(**kwargs)
        train_dataloader = self.make_dataloader(examples, **kwargs)
        torch.cuda.empty_cache()
        self.ax.eval()
        self.ax.to(self.device)
        prefix_length = kwargs["prefix_length"]

        hidden_size = self.model.config.hidden_size
        positive_sum = torch.zeros(num_positions, hidden_size, device=self.device)
        negative_sum = torch.zeros(num_positions, hidden_size, device=self.device)
        positive_count = torch.zeros(num_positions, device=self.device)
        negative_count = torch.zeros(num_positions, device=self.device)

        for _ in range(self.training_args.n_epochs):
            for batch in train_dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                activations = gather_residual_activations(
                    self.model, self.layer,
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
                ).detach().float()
                real_lengths = inputs["attention_mask"].sum(dim=1)
                labels = inputs["labels"]

                for k in range(min(num_positions, activations.shape[1])):
                    # position k back from the end; valid only if the row has that many
                    # real tokens once the chat-template prefix is excluded
                    valid = (real_lengths - prefix_length) > k
                    if not valid.any():
                        continue
                    acts_k = activations[:, activations.shape[1] - 1 - k, :]
                    is_positive = valid & (labels == 1)
                    is_negative = valid & (labels != 1)
                    positive_sum[k] += acts_k[is_positive].sum(dim=0)
                    negative_sum[k] += acts_k[is_negative].sum(dim=0)
                    positive_count[k] += is_positive.sum()
                    negative_count[k] += is_negative.sum()

        empty = ((positive_count == 0) | (negative_count == 0)).nonzero().flatten().tolist()
        if empty:
            logger.warning(
                f"No examples reached positions {empty}; their vectors will be zero. "
                f"Consider lowering num_positions ({num_positions}).")

        weight = (positive_sum / positive_count.clamp(min=1).unsqueeze(1)) - \
            (negative_sum / negative_count.clamp(min=1).unsqueeze(1))
        # normalize each position independently, so the steering_factors sweep means the
        # same thing at every position (as it does for the single-vector methods)
        eps = torch.finfo(weight.dtype).eps
        self.positional_weight = weight / (weight.norm(dim=1, keepdim=True) + eps)

        # collapsed direction, used by latent/detection mode
        collapsed = self.positional_weight.mean(dim=0, keepdim=True)
        self.ax.proj.weight.data = collapsed.to(self.ax.proj.weight.dtype)
        set_decoder_norm_to_unit_norm(self.ax)

    def save(self, dump_dir, **kwargs):
        """Save both directions into the standard {model_name}_weight.pt as a dict.

        train.py merges per-rank checkpoints by concatenating along dim 0, and already
        handles dict-valued weight files (train.py:621-626), so packing both tensors into
        one dict lets the positional weights ride the existing merge with no change to
        train.py. A separate file would silently never be merged.
        """
        model_name = kwargs.get("model_name", self.__str__())
        weight_file = dump_dir / f"{model_name}_weight.pt"
        weight = {
            "collapsed": self.ax.proj.weight.data.cpu(),                    # 1, h
            "positional": self.positional_weight.data.cpu().unsqueeze(0),   # 1, num_positions, h
        }
        if weight_file.exists():
            previous = torch.load(weight_file, weights_only=True)
            weight = {k: torch.cat([previous[k], v], dim=0) for k, v in weight.items()}
        torch.save(weight, weight_file)

        bias_file = dump_dir / f"{model_name}_bias.pt"
        bias = self.ax.proj.bias.data.cpu()
        if bias_file.exists():
            bias = torch.cat([torch.load(bias_file, weights_only=True), bias], dim=0)
        torch.save(bias, bias_file)

    def load(self, dump_dir=None, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        weight = torch.load(
            f"{dump_dir}/{model_name}_weight.pt", map_location=torch.device("cpu"), weights_only=True)
        if kwargs.get("mode") == "steering":
            # derive both dims from the checkpoint so train and inference cannot disagree
            kwargs["low_rank_dimension"] = weight["positional"].shape[0]
            kwargs["num_positions"] = weight["positional"].shape[1]
            self.make_model(**kwargs)
            self.ax.proj_weight.data = weight["positional"].to(self.device)
        else:
            bias = torch.load(
                f"{dump_dir}/{model_name}_bias.pt", map_location=torch.device("cpu"), weights_only=True)
            kwargs["low_rank_dimension"] = weight["collapsed"].shape[0]
            self.make_model(**kwargs)
            self.ax.proj.weight.data = weight["collapsed"].to(self.device)
            self.ax.proj.bias.data = bias.to(self.device)