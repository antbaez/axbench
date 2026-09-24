# inference.py: Inference with existing subspaces.
#
# example launch command:
#     torchrun --nproc_per_node=NUM_GPUS axbench/scripts/inference.py --config axbench/demo/sweep/inference.yaml --mode latent
import warnings
warnings.filterwarnings("ignore", message=r"pyreft not installed.*")
warnings.filterwarnings("ignore", message=r"HyperSteer unavailable.*")

import os, argparse, yaml, json, glob, pickle, re, time, itertools, datetime, uuid
import contextlib
import shutil
import pandas as pd
from tqdm.auto import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import atexit

from axbench.utils.dataset import (
    DatasetFactory,
    SteeringDatasetFactory,
    consolidate_steering_eval_cache
)
from axbench.utils.constants import * 
from axbench.utils.model_utils import get_prefix_length, get_suffix_length
from axbench.utils.norm_capture import SteeringNormRecorder
from axbench.scripts.args.dataset_args import DatasetArgs
from axbench.scripts.args.training_args import TrainingArgs
from transformers import set_seed

# all supported methods
import axbench
from openai import AsyncOpenAI
import httpx, asyncio

import logging
import torch.distributed as dist
import sys

# Initialize the logger
logger = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_DELAY = 1  # in seconds
STATE_FILE = "inference_state.pkl"
CONFIG_FILE = "config.json"
METADATA_FILE = "metadata.jsonl"
STEERING_WITH_SHARED_MODELS = {"HyperSteer"}
STEERING_EXCLUDE_MODELS = {"IntegratedGradients", "InputXGradients", "PromptDetection", "BoW"}
LATENT_EXCLUDE_MODELS = {"PromptSteering", "PromptBaseline", "DiReFT", "LoReFT", "LoRA", "SFT", "HyperSteer"}
LATENT_PROMPT_PREFIX = "Generate a random sentence."

def load_config(config_path):
    """
    Load metadata from a JSON lines file.
    """
    if not os.path.exists(Path(config_path) / CONFIG_FILE):
        return None
    with open(Path(config_path) / CONFIG_FILE) as f:
        d = json.load(f)
    return d


def chunk_tag(chunk):
    """Filename suffix isolating one --chunk's outputs from every other chunk's."""
    return f"_chunk{chunk}" if chunk is not None else ""


def load_state(dump_dir, mode, rank, subfolder="inference", chunk=None):
    """
    Load the state from a file if it exists.
    """
    state_path = os.path.join(
        f"{dump_dir}/{subfolder}", f"{mode}{chunk_tag(chunk)}_{STATE_FILE}_rank_{rank}")
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            return pickle.load(f)
    return None


def save_state(dump_dir, state, partition, rank, chunk=None):
    if not isinstance(dump_dir, Path):
        dump_dir = Path(dump_dir)

    dump_dir.mkdir(parents=True, exist_ok=True)
    # Save state
    state_path = os.path.join(
        dump_dir, f"{partition}{chunk_tag(chunk)}_{STATE_FILE}_rank_{rank}")
    with open(state_path, "wb") as f:
        pickle.dump(state, f)


def load_metadata_flatten(metadata_path):
    """
    Load flatten metadata from a JSON lines file.
    """
    metadata = []
    with open(Path(metadata_path) / METADATA_FILE, 'r') as f:
        for line in f:
            data = json.loads(line)
            concept, ref = data["concept"], data["ref"]
            concept_genres_map = data["concept_genres_map"][concept]
            ref = data["ref"]
            flatten_data = {
                "concept": concept,
                "ref": ref,
                "concept_genres_map": {concept: concept_genres_map},
                "concept_id": data["concept_id"]
            }
            metadata += [flatten_data]  # Return the metadata as is
    return metadata


def save(
    dump_dir, partition,
    current_df, rank, chunk=None):
    # This function saves DataFrames per rank per partition (latent or steering)
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    # Save DataFrame
    df_path = os.path.join(
        dump_dir, f"rank_{rank}_{partition}{chunk_tag(chunk)}_data.parquet")
    
    if "defense" in current_df.columns:
        current_df["defense"] = current_df["defense"].apply(lambda x: "["+','.join(x)+"]" if isinstance(x, list) else str(x))
        
    if os.path.exists(df_path):
        existing_df = pd.read_parquet(df_path)
        
        if "defense" in existing_df.columns:
            # Convert defense column to string format if it exists
            existing_df["defense"] = existing_df["defense"].apply(lambda x: "["+','.join(x)+"]" if isinstance(x, list) else str(x))
        combined_df = pd.concat([existing_df, current_df], ignore_index=True)
    else:
        combined_df = current_df

    combined_df.to_parquet(df_path, engine='pyarrow')


def partition_concept_ids(concept_ids, world_size):
    concept_ids_per_rank = []
    n = len(concept_ids)
    chunk_size = n // world_size
    remainder = n % world_size
    start = 0
    for i in range(world_size):
        end = start + chunk_size + (1 if i < remainder else 0)
        concept_ids_per_rank.append(concept_ids[start:end])
        start = end
    return concept_ids_per_rank


def chunk_bounds(chunk, num_chunks, n_concepts):
    """Half-open [lo, hi) concept-id window for one chunk of a num_chunks-way split.

    Remainder concepts go to the leading chunks, so the windows always tile the whole
    id space exactly -- no concept can fall between two chunks and be silently dropped.
    """
    if chunk is None:
        return 0, float("inf")
    if not 0 <= chunk < num_chunks:
        raise ValueError(f"--chunk {chunk} is out of range for --num_chunks {num_chunks}")
    base, remainder = divmod(n_concepts, num_chunks)
    lo = chunk * base + min(chunk, remainder)
    hi = lo + base + (1 if chunk < remainder else 0)
    return lo, hi


def create_data_latent(dataset_factory, metadata, concept_id, num_of_examples, args):
    # prepare concept related data.
    concept = metadata[concept_id]["concept"]
    sae_link = metadata[concept_id]["ref"]
    sae_id = int(sae_link.split("/")[-1]) 
    concept_genres_map = metadata[concept_id]["concept_genres_map"]
    _, eval_contrast_concepts_map = \
        dataset_factory.prepare_concepts(
            [concept], 
            concept_genres_map=concept_genres_map,
            contrast_concepts_map={}, api_tag="inference")
    current_df = dataset_factory.create_eval_df(
        [concept], num_of_examples, concept_genres_map, {},
        eval_contrast_concepts_map, input_length=args.input_length, 
        output_length=args.output_length, concept_id=concept_id
    )
    current_df["concept_id"] = concept_id
    current_df["sae_link"] = sae_link
    current_df["sae_id"] = sae_id
    return current_df


def create_data_steering(
    dataset_factory, metadata, concept_id, num_of_examples, 
    n_steering_factors, steering_datasets, args, generate_args):

    # prepare concept related data.
    concept = metadata[concept_id]["concept"]
    sae_link = metadata[concept_id]["ref"]
    try:
        sae_id = int(sae_link.split("/")[-1]) 
    except:
        sae_id = 0

    current_df = dataset_factory.create_eval_df(
        [concept], num_of_examples, n_steering_factors, steering_datasets, concept_id=concept_id,
        steering_model_name=args.steering_model_name, steer_data_type=generate_args.steer_data_type,
        n_shots=args.n_shot, defense=args.defense, dump_dir=args.dump_dir, multishot_factors_parquet=args.multishot_factors_parquet,
        suppress_eval_dir=args.suppress_eval_dir
    )
    current_df["concept_id"] = concept_id
    current_df["sae_link"] = sae_link
    current_df["sae_id"] = sae_id

    return current_df, (concept_id, sae_link, sae_id)


def prepare_df(current_df, tokenizer, is_chat_model, model_name):
    suffix_length, _ = get_suffix_length(tokenizer)
    if is_chat_model:
        if model_name == "meta-llama/Llama-3.1-8B-Instruct":
            def apply_chat_template(row):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."}, 
                    {"role": "user", "content": row["input"]},
                    {"role": "assistant", "content": row["output"]}
                ]
                tokens = tokenizer.apply_chat_template(messages, tokenize=True)[1:-suffix_length]
                return tokenizer.decode(tokens)
            current_df['input'] = current_df.apply(apply_chat_template, axis=1)
        else:
            def apply_chat_template(row):
                messages = [
                    {"role": "user", "content": row["input"]},
                    {"role": "assistant", "content": row["output"]}
                ]
                tokens = tokenizer.apply_chat_template(messages, tokenize=True)[1:-suffix_length]
                return tokenizer.decode(tokens)
            current_df['input'] = current_df.apply(apply_chat_template, axis=1)
    return current_df


def infer_steering(args, rank, world_size, device, logger, training_args, generate_args, suppress_eval_dir=None):
    data_dir = args.data_dir
    train_dir = args.train_dir
    dump_dir = args.dump_dir
    overwrite_inference_dump_dir = Path(args.overwrite_inference_dump_dir) if args.overwrite_inference_dump_dir is not None else Path(dump_dir) / "inference"
    num_of_examples = args.steering_num_of_examples
    config = load_config(train_dir)
    metadata = load_metadata_flatten(data_dir)
    layer = int(args.steering_layer) if args.steering_layer is not None else config["layer"] if config else 0  # default layer for prompt baselines
    steering_layers = args.steering_layers if args.steering_layers is not None else [layer]
    steering_factors = args.steering_factors
    steering_datasets = args.steering_datasets

    chunk = getattr(args, "chunk", None)
    state = load_state(args.dump_dir, "steering", rank, chunk=chunk)
    last_concept_id_processed = state.get("last_concept_id", None) if state else None
    logger.warning(f"Rank {rank} last concept_id processed: {last_concept_id_processed}")

    # Get list of all concept_ids
    concept_ids = [metadata[i]["concept_id"] for i in range(len(metadata))]

    # Partition concept_ids among ranks sequentially
    concept_ids_per_rank = partition_concept_ids(concept_ids, world_size)
    my_concept_ids = concept_ids_per_rank[rank]

    # Restrict to this chunk's concept window, so N independent single-GPU jobs
    # cover disjoint concepts (rank partitioning is a no-op at world_size=1).
    if chunk is not None:
        chunk_lo, chunk_hi = chunk_bounds(chunk, args.num_chunks, max(concept_ids) + 1)
        my_concept_ids = [c for c in my_concept_ids if chunk_lo <= c < chunk_hi]
        logger.warning(
            f"Chunk {chunk}/{args.num_chunks}: concepts [{chunk_lo}, {chunk_hi}) "
            f"-> {len(my_concept_ids)} to process")
        # Chunk files and their resume state are deleted once merged, so a requeue
        # after the merge would otherwise find no state and redo the whole window.
        window_ids = [c for c in concept_ids if chunk_lo <= c < chunk_hi]
        if _chunk_already_merged(args, "steering", window_ids):
            logger.warning(
                f"Chunk {chunk}: all {len(window_ids)} concepts already in the merged "
                f"steering_data.parquet. Exiting.")
            return

    if last_concept_id_processed is not None:
        if last_concept_id_processed in my_concept_ids:
            idx = my_concept_ids.index(last_concept_id_processed)
            my_concept_ids = my_concept_ids[idx+1:]
        else:
            # If last_concept_id_processed is not in my_concept_ids, process all
            pass

    if len(my_concept_ids) == 0:

        # Synchronize all processes
        dist.barrier()

        # Rank 0 merges results
        if rank == 0 and chunk is None:
            logger.warning("Rank 0 is merging results.")
            # Merge per-rank results
            all_parquet_files = list((overwrite_inference_dump_dir).glob("rank_*_steering_data.parquet"))
            # Parse filenames to extract rank
            import re
            pattern = re.compile(r'rank_(\d+)_steering_data\.parquet')

            file_info_list = []
            for parquet_file in all_parquet_files:
                match = pattern.match(parquet_file.name)
                if match:
                    rank_str = match.group(1)
                    rank_int = int(rank_str)
                    file_info_list.append({
                        'rank': rank_int,
                        'file': parquet_file
                    })
                else:
                    logger.warning(f"Filename {parquet_file.name} does not match the expected pattern.")

            # Sort the file_info_list by rank
            file_info_list.sort(key=lambda x: x['rank'])

            # Read and concatenate dataframes
            dfs = []
            for info in file_info_list:
                df = pd.read_parquet(info['file'])
                dfs.append(df)
            if len(dfs) > 0:
                combined_df = pd.concat(dfs, ignore_index=True)
                # Optionally sort combined_df by 'concept_id' if needed
                combined_df = combined_df.sort_values(by=['concept_id', 'input_id', 'factor']).reset_index(drop=True)
                combined_df.to_parquet(overwrite_inference_dump_dir / "steering_data.parquet", engine='pyarrow')
                logger.warning(f"Saved combined steering inference results to {overwrite_inference_dump_dir / 'steering_data.parquet'}")
            else:
                logger.warning("No results to merge.")

            # Optionally, delete per-rank files
            for info in file_info_list:
                os.remove(info['file'])
                logger.warning(f"Deleted {info['file']}")

        logger.warning(f"Rank {rank} has no concepts to process. Exiting.")
        return

    # Create a new OpenAI client.
    lm_client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=60.0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=100,
                max_connections=1000
            ),
            headers={"Connection": "close"},
        ),
        max_retries=3,
    )

    # Initialize the dataset factory with the tokenizer.
    if "google/gemma-3" in args.steering_model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.steering_model_name, use_fast=False, model_max_length=128000)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.steering_model_name, use_fast=False, model_max_length=1024)
    tokenizer.padding_side = "right"
    if "PromptSteering" in args.models:
        has_prompt_steering = True
    else:
        if "LsReFT" in args.models and training_args.models["LsReFT"].use_synergy:
            has_prompt_steering = True
        else:
            has_prompt_steering = False
    dataset_factory = SteeringDatasetFactory(
        tokenizer, dump_dir,
        steering_instructions_dist=getattr(args, "steering_instructions_dist", "train"),
        random_concept_injection_n_steps=getattr(args, "random_concept_injection_n_steps", 2),
        master_data_dir=args.master_data_dir, lm_client=lm_client,
        lm_model=args.lm_model, use_cache=args.lm_use_cache,
        has_prompt_steering=has_prompt_steering
    )
    is_chat_model = True if args.model_name in CHAT_MODELS else False
    prefix_length = 1 # prefix is default to 1 for all models due to the BOS token.
    if is_chat_model:
        prefix_length = get_prefix_length(tokenizer)
        logger.warning(f"Chat model prefix length: {prefix_length}")
        
    # Load model instance onto device
    if args.use_bf16:
        logger.warning(f"Using bfloat16 for model {args.model_name}")
    if "gemma-3" in args.model_name:
        from transformers import Gemma3ForCausalLM
        model_instance = Gemma3ForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16 if args.use_bf16 else None, device_map=device)
    else:
        model_instance = AutoModelForCausalLM.from_pretrained(
            args.steering_model_name if args.steering_model_name else args.model_name, 
            torch_dtype=torch.bfloat16 if args.use_bf16 else None, device_map=device
        )
    model_instance = model_instance.eval()

    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        need_resize = True
    else:
        need_resize = False
    if need_resize:
        model_instance.resize_token_embeddings(len(tokenizer))

    # Prepare data per concept
    data_per_concept = {}
    for concept_id in my_concept_ids:
        
        """current_df = pd.read_parquet(f"/nlp/scr/sjd24/merge/axbench/axbench/concept10/prod_2b_l20_v1/debug_data/{concept_id}_steering_data.parquet")
        sae_link, sae_id = None, None"""
        
        current_df, (_, sae_link, sae_id) = create_data_steering(
            dataset_factory, metadata, concept_id, num_of_examples,
            steering_factors, steering_datasets, args, generate_args
        )
        data_per_concept[concept_id] = (current_df, sae_link, sae_id)

    # Preview dump: base prompts actually fed to the model for the first 10 concepts
    # (by concept_id, not just this shard's first 10), so a run's prompts can be
    # sanity-checked without digging through the full steering_data.parquet.
    first_ten_concepts = [c for c in concept_ids[:10] if c in data_per_concept]
    if first_ten_concepts:
        base_prompts_preview = {}
        for concept_id in first_ten_concepts:
            current_df, _, _ = data_per_concept[concept_id]
            preview_rows = current_df.drop_duplicates(subset="input_id").sort_values("input_id")[:10]
            base_prompts_preview[str(concept_id)] = [
                {
                    "prompt": row["input"],
                    "contrast_concept": row["contrast_concept"] if "contrast_concept" in current_df.columns else None,
                }
                for _, row in preview_rows.iterrows()
            ]
        preview_path = Path(overwrite_inference_dump_dir) / f"base_prompts_preview{chunk_tag(chunk)}_rank{rank}.json"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        with open(preview_path, "w") as f:
            json.dump(base_prompts_preview, f, indent=2)
        logger.warning(f"Wrote base prompts for concepts {first_ten_concepts} to {preview_path}")

    # Preview dump: prompt + every model's generation, for the first 10 input_ids x
    # every steering factor, for the first 10 concepts -- written incrementally as
    # each of those concepts finishes (not batched at the end), so a preempted run
    # keeps whatever it already produced. Pre-loaded from disk so a resumed run
    # (which skips concepts already done in an earlier attempt, see
    # last_concept_id_processed above) doesn't lose entries a prior attempt wrote.
    sample_generations_path = (
        Path(overwrite_inference_dump_dir) / f"sample_generations_preview{chunk_tag(chunk)}_rank{rank}.json")
    sample_generations_preview = {}
    if first_ten_concepts and sample_generations_path.exists():
        with open(sample_generations_path) as f:
            sample_generations_preview = json.load(f)

    # Preload models that are shared across concepts, like HyperSteer.
    preloaded_models = dict()
    for model_name in training_args.models.keys():
        if model_name in STEERING_EXCLUDE_MODELS:
            continue
        if model_name in STEERING_WITH_SHARED_MODELS:
            model_class = getattr(axbench, model_name)
            logger.info(f"Loading {model_class} on {device}.")
            
            benchmark_model = model_class(
                model_instance, tokenizer, layer=layer,
                low_rank_dimension=len(metadata),
                device=device,
                training_args=training_args.models[model_name],
                lm_model_name=training_args.model_name,
            )
            benchmark_model.load(
                dump_dir=train_dir, low_rank_dimension=1, mode="steering", 
                hypernet_initialize_from_pretrained=training_args.models[model_name].hypernet_initialize_from_pretrained,
                hypernet_name_or_path=training_args.models[model_name].hypernet_name_or_path,
                num_hidden_layers=training_args.models[model_name].num_hidden_layers,
            )
            preloaded_models[model_name] = benchmark_model

    # capture_norms: record the layer-`layer` residual norm at every token of every
    # steered forward pass, straight from the intervention hook (no extra inference).
    # One parquet per concept under inference/norms/, written as each concept finishes.
    capture_norms = bool(getattr(args, "capture_norms", False))
    norms_dir = Path(overwrite_inference_dump_dir) / "norms"
    if capture_norms:
        norms_dir.mkdir(parents=True, exist_ok=True)
        logger.warning(f"capture_norms is on: writing per-token layer-{layer} norms to {norms_dir}")

    # Now loop over concept_ids and use preloaded models
    for concept_id in my_concept_ids:
        current_df, sae_link, sae_id = data_per_concept[concept_id]
        concept_norm_records = []
        for model_name in args.models:
            if model_name in STEERING_EXCLUDE_MODELS:
                continue
            
            if model_name not in STEERING_WITH_SHARED_MODELS:
                model_class = getattr(axbench, model_name)
                logger.info(f"Loading {model_class} on {device}.")

                benchmark_model = model_class(
                    model_instance, tokenizer, layer=layer,
                    training_args=training_args.models[model_name] if model_name not in {"PromptSteering", "GemmaScopeSAE"} else None, # we init with training args as well
                    low_rank_dimension=len(metadata),
                    device=device, steering_layers=steering_layers,
                )
                if model_name in {"PromptSteering", "GemmaScopeSAE"}:
                    lr = 1
                else:
                    lr = training_args.models[model_name].low_rank_dimension if training_args.models[model_name].low_rank_dimension else 1
                benchmark_model.load(
                    dump_dir=train_dir, sae_path=metadata[0]["ref"], 
                    mode="steering",
                    priority_mode="compute_priority",
                    intervention_type=args.steering_intervention_type,
                    concept_id=concept_id,
                    low_rank_dimension=lr
                )
                benchmark_model.to(device)
                if hasattr(benchmark_model, 'ax') and args.use_bf16:
                    if model_name not in {"PreferenceLoReFT", "ConceptLoReFT",}:
                        if isinstance(benchmark_model.ax, list):
                            for ax in benchmark_model.ax:
                                ax.eval()
                            ax.to(torch.bfloat16)
                        else:
                            benchmark_model.ax.eval()
                            benchmark_model.ax.to(torch.bfloat16)
            else:
                benchmark_model = preloaded_models[model_name]
                
                benchmark_model.to(device)
                if hasattr(benchmark_model, 'ax') and args.use_bf16:
                    benchmark_model.ax.eval()
                    benchmark_model.ax.to(torch.bfloat16)
                
            # Pre-compute mean activations once
            if model_name not in {"LoReFT", "BoW"} and model_name not in LATENT_EXCLUDE_MODELS:
                benchmark_model.pre_compute_mean_activations(
                    os.path.join(dump_dir, "inference"),
                    master_data_dir=args.master_data_dir,
                    disable_neuronpedia_max_act=args.disable_neuronpedia_max_act,
                    metadata=metadata,
                )
                # predict_steer looks up max_act per row and falls back to 1.0 on a miss,
                # so a latent run covering only some chunks would silently rescale just
                # part of the sweep. Same key column predict_steer resolves (model.py).
                key_col = ("sae_id" if "sae" in model_name.lower()
                           and not args.disable_neuronpedia_max_act else "concept_id")
                uncovered = sorted(
                    set(current_df[key_col]) - set(benchmark_model.max_activations))
                if uncovered:
                    raise ValueError(
                        f"{model_name}: latent_data.parquet has no max_act for "
                        f"{key_col}(s) {uncovered[:10]} ({len(uncovered)} total) -- these "
                        f"would steer at factor x 1.0. Merge every latent chunk first.")
            unique_concept_ids = list(set(current_df["concept_id"].tolist()))
            logger.warning(f"Inference steering with {model_name} on {device} for concept {concept_id}.")
            norm_recorder = None
            if capture_norms:
                ax = getattr(benchmark_model, "ax", None)
                if ax is None or isinstance(ax, list) or not hasattr(benchmark_model, "ax_model"):
                    logger.warning(f"capture_norms: {model_name} has no single pyvene "
                                   f"intervention to hook; skipping its norms.")
                else:
                    # the verification prints run once per model, on this rank's first concept
                    norm_recorder = SteeringNormRecorder(
                        benchmark_model.model, ax, benchmark_model.layer,
                        model_name=model_name, verify=(concept_id == my_concept_ids[0]),
                        logger=logger)
            # Run prediction; the recorder's hooks are removed on exit even if this raises
            with norm_recorder if norm_recorder is not None else contextlib.nullcontext():
                results = benchmark_model.predict_steer(
                    current_df, concept_id=unique_concept_ids[0] if len(unique_concept_ids) == 1 else unique_concept_ids, sae_link=None, sae_id=None,
                    batch_size=int(args.steering_batch_size),
                    eval_output_length=int(args.steering_output_length),
                    temperature=float(args.temperature),
                    prefix_length=prefix_length,
                    positions=training_args.models[model_name].intervention_positions if model_name not in {"PromptSteering", "GemmaScopeSAE"} else None,
                    use_synergy=False,
                    disable_neuronpedia_max_act=args.disable_neuronpedia_max_act,
                    intervene_on_prompt=args.intervene_on_prompt if args.intervene_on_prompt is not None else True,
                    return_vector=False,
                    norm_recorder=norm_recorder,
                )
            if norm_recorder is not None:
                if not norm_recorder.records:
                    logger.warning(f"capture_norms: {model_name} recorded nothing -- its "
                                   f"predict_steer override does not call the recorder.")
                concept_norm_records.extend(norm_recorder.records)
            # Store the results in current_df
            for k, v in results.items():
                current_df[f"{model_name}_{k}"] = v
                
            if model_name not in STEERING_WITH_SHARED_MODELS:
                del benchmark_model
            else:
                benchmark_model = benchmark_model.to("cpu") # move shared model to cpu to save memory
                
            torch.cuda.empty_cache()

        if concept_id in first_ten_concepts:
            model_names = [m for m in args.models if m not in STEERING_EXCLUDE_MODELS]
            first_ten_input_ids = sorted(current_df["input_id"].unique())[:10]
            rows = current_df[current_df["input_id"].isin(first_ten_input_ids)].sort_values(
                ["input_id", "factor"])
            sample_generations_preview[str(concept_id)] = {
                "concept": rows["input_concept"].iloc[0],
                "generations": [
                    {
                        "input_id": int(row["input_id"]),
                        "factor": float(row["factor"]),
                        "base_instruction": row["base_instruction"] if "base_instruction" in current_df.columns else None,
                        "contrast_concept": row["contrast_concept"] if "contrast_concept" in current_df.columns else None,
                        "prompt": row["original_prompt"],
                        "generations": {
                            m: row[f"{m}_steered_generation"] for m in model_names
                            if f"{m}_steered_generation" in current_df.columns
                        },
                    }
                    for _, row in rows.iterrows()
                ],
            }
            with open(sample_generations_path, "w") as f:
                json.dump(sample_generations_preview, f, indent=2, ensure_ascii=False)
            logger.warning(
                f"Wrote sample generations for concept {concept_id} to {sample_generations_path}")

        # written before the resume state below, so a preemption in between redoes this
        # concept rather than skipping it with its norms missing
        if capture_norms and concept_norm_records:
            norms_path = norms_dir / f"concept{concept_id}_rank{rank}{chunk_tag(chunk)}.parquet"
            pd.DataFrame(concept_norm_records).to_parquet(norms_path, engine="pyarrow")
            logger.warning(
                f"Saved {len(concept_norm_records)} norm records for concept {concept_id} "
                f"to {norms_path}")

        save(overwrite_inference_dump_dir, 'steering', current_df, rank, chunk=chunk)
        logger.warning(
            f"Saved inference results for concept {concept_id} to "
            f"rank_{rank}_steering{chunk_tag(chunk)}_data.parquet")
        # After processing, save state
        current_state = {'last_concept_id': concept_id}
        save_state(overwrite_inference_dump_dir, current_state, 'steering', rank, chunk=chunk)

    # Synchronize all processes
    dist.barrier()

    # Rank 0 merges results. Skipped when chunked: each job would otherwise merge
    # only its own shard over the combined file and then delete its inputs. Instead,
    # once every chunk's output is present, the last chunk to finish auto-merges them.
    if rank == 0 and chunk is not None:
        maybe_auto_merge_chunks(args, "steering", logger)
    if rank == 0 and chunk is None:
        logger.warning("Rank 0 is merging results.")
        # Merge per-rank results
        all_parquet_files = list((Path(dump_dir) / "inference").glob("rank_*_steering_data.parquet"))
        # Parse filenames to extract rank
        import re
        pattern = re.compile(r'rank_(\d+)_steering_data\.parquet')

        file_info_list = []
        for parquet_file in all_parquet_files:
            match = pattern.match(parquet_file.name)
            if match:
                rank_str = match.group(1)
                rank_int = int(rank_str)
                file_info_list.append({
                    'rank': rank_int,
                    'file': parquet_file
                })
            else:
                logger.warning(f"Filename {parquet_file.name} does not match the expected pattern.")

        # Sort the file_info_list by rank
        file_info_list.sort(key=lambda x: x['rank'])

        # Read and concatenate dataframes
        dfs = []
        for info in file_info_list:
            df = pd.read_parquet(info['file'])
            dfs.append(df)
        if len(dfs) > 0:
            combined_df = pd.concat(dfs, ignore_index=True)
            # Optionally sort combined_df by 'concept_id' if needed
            combined_df = combined_df.sort_values(by=['concept_id', 'input_id', 'factor']).reset_index(drop=True)
            combined_df.to_parquet(Path(dump_dir) / "inference" / "steering_data.parquet", engine='pyarrow')
            logger.warning(f"Saved combined steering inference results to {Path(dump_dir) / 'inference' / 'steering_data.parquet'}")
            consolidate_steering_eval_cache(
                dump_dir, logger, getattr(args, "steering_instructions_dist", "train"))
        else:
            logger.warning("No results to merge.")

        # Optionally, delete per-rank files
        for info in file_info_list:
            os.remove(info['file'])
            logger.warning(f"Deleted {info['file']}")


def infer_latent(args, rank, world_size, device, logger, training_args, generate_args):
    data_dir = args.data_dir
    train_dir = args.train_dir
    dump_dir = Path(args.dump_dir) / "inference"
    num_of_examples = args.latent_num_of_examples
    config = load_config(train_dir)
    metadata = load_metadata_flatten(data_dir)
    layer = config["layer"] if config else 0  # default layer for prompt baselines

    chunk = getattr(args, "chunk", None)
    state = load_state(args.dump_dir, "latent", rank, chunk=chunk)
    last_concept_id_processed = state.get("last_concept_id", None) if state else None
    logger.warning(f"Rank {rank} last concept_id processed: {last_concept_id_processed}")

    # Get list of all concept_ids
    concept_ids = [metadata[i]["concept_id"] for i in range(len(metadata))]

    # Partition concept_ids among ranks sequentially
    concept_ids_per_rank = partition_concept_ids(concept_ids, world_size)
    my_concept_ids = concept_ids_per_rank[rank]

    # Restrict to this chunk's concept window, so N independent single-GPU jobs
    # cover disjoint concepts (rank partitioning is a no-op at world_size=1).
    if chunk is not None:
        chunk_lo, chunk_hi = chunk_bounds(chunk, args.num_chunks, max(concept_ids) + 1)
        my_concept_ids = [c for c in my_concept_ids if chunk_lo <= c < chunk_hi]
        logger.warning(
            f"Chunk {chunk}/{args.num_chunks}: concepts [{chunk_lo}, {chunk_hi}) "
            f"-> {len(my_concept_ids)} to process")
        # Chunk files and their resume state are deleted once merged, so a requeue
        # after the merge would otherwise find no state and redo the whole window.
        window_ids = [c for c in concept_ids if chunk_lo <= c < chunk_hi]
        if _chunk_already_merged(args, "latent", window_ids):
            logger.warning(
                f"Chunk {chunk}: all {len(window_ids)} concepts already in the merged "
                f"latent_data.parquet. Exiting.")
            return

    if last_concept_id_processed is not None:
        if last_concept_id_processed in my_concept_ids:
            idx = my_concept_ids.index(last_concept_id_processed)
            my_concept_ids = my_concept_ids[idx+1:]
        else:
            # If last_concept_id_processed is not in my_concept_ids, process all
            pass

    if len(my_concept_ids) == 0:
        logger.warning(f"Rank {rank} has no concepts to process. Exiting.")
        return

    # Create a new OpenAI client.
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=60.0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=100,
                max_connections=1000
            ),
            headers={"Connection": "close"},
        ),
        max_retries=3,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, model_max_length=1024)
    tokenizer.padding_side = "right"

    # Load model instance onto device
    if args.use_bf16:
        logger.warning(f"Using bfloat16 for model {args.model_name}")
    model_instance = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        torch_dtype=torch.bfloat16 if args.use_bf16 else None, 
        device_map=device
    )
    is_chat_model = True if args.model_name in CHAT_MODELS else False
    model_instance = model_instance.eval()

    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        need_resize = True
    else:
        need_resize = False
    if need_resize:
        model_instance.resize_token_embeddings(len(tokenizer))

    prefix_length = 1 # prefix is default to 1 for all models due to the BOS token.
    if is_chat_model:
        prefix_length = get_prefix_length(tokenizer)
        logger.warning(f"Chat model prefix length: {prefix_length}")

    # Load dataset factory for evals.
    dataset_factory = DatasetFactory(
        None, client, tokenizer, generate_args.dataset_category, None, None, args.dump_dir,
        use_cache=False, master_data_dir=args.master_data_dir,
        lm_model=args.lm_model, logger=logger, is_inference=True,
        overwrite_inference_data_dir=training_args.overwrite_inference_data_dir
    )
    atexit.register(dataset_factory.save_cache)
    atexit.register(dataset_factory.reset_stats)

    has_latent_model = False
    for model_name in args.models:
        # load model on the fly to save memory
        if model_name not in LATENT_EXCLUDE_MODELS:
            has_latent_model = True
            break

    if not has_latent_model:
        logger.warning("No latent model to infer. Exiting.")
        return

    # Now loop over concept_ids and use preloaded models
    cache_df = {}
    for concept_id in my_concept_ids:
        for model_name in args.models:
            # load model on the fly to save memory
            if model_name in LATENT_EXCLUDE_MODELS:
                continue
            model_class = getattr(axbench, model_name)
            logger.info(f"Loading {model_class} on {device}.")
            benchmark_model = model_class(
                model_instance, tokenizer, layer=layer,
                low_rank_dimension=len(metadata),
                device=device
            )
            benchmark_model.load(
                dump_dir=train_dir, sae_path=metadata[0]["ref"], mode="latent",
                concept_id=concept_id
            )
            benchmark_model.to(device)
            if hasattr(benchmark_model, 'ax') and args.use_bf16:
                benchmark_model.ax.eval()
                benchmark_model.ax.to(torch.bfloat16)

            dataset_category = generate_args.dataset_category
            if (concept_id, dataset_category) not in cache_df:
                current_df = create_data_latent(
                    dataset_factory, metadata, concept_id, num_of_examples, args)
                logger.warning(f"Inference latent with {model_name} on {device} for concept {concept_id}.")
                current_df = prepare_df(current_df, tokenizer, is_chat_model, args.model_name)
                cache_df[(concept_id, dataset_category)] = current_df
            else:
                current_df = cache_df[(concept_id, dataset_category)]

            results = benchmark_model.predict_latent(
                current_df, batch_size=args.latent_batch_size, prefix_length=prefix_length
            )
            # Store the results in current_df
            for k, v in results.items():
                if k == "tokens":
                    if "tokens" not in current_df:
                        current_df["tokens"] = v  # for tokens, they are global
                    else:
                        continue
                else:
                    current_df[f"{model_name}_{k}"] = v
            del benchmark_model
            torch.cuda.empty_cache()
        save(dump_dir, 'latent', current_df, rank, chunk=chunk)
        logger.warning(
            f"Saved inference results for concept {concept_id} to "
            f"rank_{rank}_latent{chunk_tag(chunk)}_data.parquet")
        # After processing, save state. Written into `dump_dir` (= <root>/inference),
        # which is where load_state reads from -- previously this wrote to the dump
        # root instead, so latent mode could never actually resume after preemption.
        current_state = {'last_concept_id': concept_id}
        save_state(dump_dir, current_state, 'latent', rank, chunk=chunk)

    # Synchronize all processes
    dist.barrier()

    # Rank 0 merges results. Skipped when chunked: each job would otherwise merge
    # only its own shard over the combined file and then delete its inputs. Instead,
    # once every chunk's output is present, the last chunk to finish auto-merges them.
    if rank == 0 and chunk is not None:
        maybe_auto_merge_chunks(args, "latent", logger)
    if rank == 0 and chunk is None:
        logger.warning("Rank 0 is merging results.")
        # Merge per-rank results
        all_parquet_files = list(dump_dir.glob("rank_*_latent_data.parquet"))
        # Parse filenames to extract rank
        import re
        pattern = re.compile(r'rank_(\d+)_latent_data\.parquet')

        file_info_list = []
        for parquet_file in all_parquet_files:
            match = pattern.match(parquet_file.name)
            if match:
                rank_str = match.group(1)
                rank_int = int(rank_str)
                file_info_list.append({
                    'rank': rank_int,
                    'file': parquet_file
                })
            else:
                logger.warning(f"Filename {parquet_file.name} does not match the expected pattern.")

        # Sort the file_info_list by rank
        file_info_list.sort(key=lambda x: x['rank'])

        # Read and concatenate dataframes
        dfs = []
        for info in file_info_list:
            df = pd.read_parquet(info['file'])
            dfs.append(df)
        if len(dfs) > 0:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df.to_parquet(dump_dir / "latent_data.parquet", engine='pyarrow')
            logger.warning(f"Saved combined latent inference results to {dump_dir / 'latent_data.parquet'}")
        else:
            logger.warning("No results to merge.")

        # Optionally, delete per-rank files
        for info in file_info_list:
            os.remove(info['file'])
            logger.warning(f"Deleted {info['file']}")

        # Save top logits (optional)
        logger.warning("Saving top logits...")
        if "LsReFT" in args.models:
            model_name = "LsReFT"
            model_class = getattr(axbench, model_name)
            benchmark_model = model_class(
                model_instance, tokenizer, layer=layer,
                low_rank_dimension=len(metadata),
                device=device
            )
            benchmark_model.load(dump_dir=train_dir, sae_path=metadata[0]["ref"])
            if hasattr(benchmark_model, 'ax') and args.use_bf16:
                benchmark_model.ax.eval()
                benchmark_model.ax.to(torch.bfloat16)
            benchmark_model.to(device)
            for concept_id in concept_ids:
                top_logits, neg_logits = benchmark_model.get_logits(concept_id, k=10)
                top_logits_entry = {
                    "concept_id": int(concept_id),
                    "results": {
                        model_name: {
                            "top_logits": top_logits,
                            "neg_logits": neg_logits
                        }
                    }
                }
                with open(dump_dir / "top_logits.jsonl", "a") as f:
                    f.write(json.dumps(top_logits_entry) + "\n")


def infer_latent_imbalance(args, rank, world_size, device, logger, training_args, generate_args):
    data_dir = args.data_dir
    train_dir = args.train_dir
    dump_dir = args.dump_dir
    num_of_examples = args.latent_num_of_examples
    config = load_config(train_dir)
    metadata = load_metadata_flatten(data_dir)
    layer = config["layer"] if config else 0  # default layer for prompt baselines

    # Get list of all concept_ids
    concept_ids = [metadata[i]["concept_id"] for i in range(len(metadata))]

    # Create a new OpenAI client.
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=60.0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=100,
                max_connections=1000
            ),
            headers={"Connection": "close"},
        ),
        max_retries=3,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, model_max_length=1024)
    tokenizer.padding_side = "right"

    # Load model instance onto device
    if args.use_bf16:
        logger.warning(f"Using bfloat16 for model {args.model_name}")
    model_instance = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        torch_dtype=torch.bfloat16 if args.use_bf16 else None, 
        device_map=device
    )
    is_chat_model = True if args.model_name in CHAT_MODELS else False
    model_instance = model_instance.eval()

    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        need_resize = True
    else:
        need_resize = False
    if need_resize:
        model_instance.resize_token_embeddings(len(tokenizer))

    prefix_length = 1 # prefix is default to 1 for all models due to the BOS token.
    if is_chat_model:
        prefix_length = get_prefix_length(tokenizer)
        logger.warning(f"Chat model prefix length: {prefix_length}")

    # Load dataset factory for evals.
    dataset_factory = DatasetFactory(
        None, client, tokenizer, generate_args.dataset_category, None, None, dump_dir,
        use_cache=False, master_data_dir=args.master_data_dir,
        lm_model=args.lm_model, logger=logger, is_inference=True,
        overwrite_inference_data_dir=training_args.overwrite_inference_data_dir
    )
    atexit.register(dataset_factory.save_cache)
    atexit.register(dataset_factory.reset_stats)

    has_latent_model = False
    for model_name in args.models:
        # load model on the fly to save memory
        if model_name not in LATENT_EXCLUDE_MODELS:
            has_latent_model = True
            break

    if not has_latent_model:
        logger.warning("No latent model to infer. Exiting.")
        return

    logger.warning(f"We are inferencing imbalanced latent once for all concepts with factor {args.imbalance_factor}.")
    all_negative_df = dataset_factory.create_imbalance_eval_df(
        num_of_examples, factor=args.imbalance_factor)
    all_negative_df = prepare_df(all_negative_df, tokenizer, is_chat_model, args.model_name)

    # save all_negative_df to disk
    dump_dir = Path(dump_dir) / "inference_imbalance"
    dump_dir.mkdir(parents=True, exist_ok=True)
    all_negative_df.to_parquet(Path(dump_dir) / "all_negative_df.parquet", engine='pyarrow')

    for model_name in args.models:
        # load model on the fly to save memory
        if model_name in LATENT_EXCLUDE_MODELS:
            continue
        model_class = getattr(axbench, model_name)
        logger.warning(f"Loading {model_class} on {device}.")
        benchmark_model = model_class(
            model_instance, tokenizer, layer=layer,
            low_rank_dimension=len(metadata),
            device=device
        )
        if model_name in {"PromptDetection", "BoW"}:
            for concept_id in concept_ids:
                benchmark_model.load(
                    dump_dir=train_dir, sae_path=metadata[0]["ref"], mode="latent",
                    concept_id=concept_id
                )
                benchmark_model.to(device)
                if hasattr(benchmark_model, 'ax') and args.use_bf16:
                    benchmark_model.ax.eval()
                    benchmark_model.ax.to(torch.bfloat16)
                results = benchmark_model.predict_latent(
                    all_negative_df, 
                    batch_size=args.latent_batch_size, 
                    prefix_length=prefix_length,
                    concept=metadata[concept_id]["concept"],
                )
                # save results to disk
                with open(dump_dir / f"{model_name}_concept_{concept_id}_latent_results.pkl", "wb") as f:
                    pickle.dump(results, f)
        else:
            benchmark_model.load(
                dump_dir=train_dir, sae_path=metadata[0]["ref"], mode="latent"
            )
            benchmark_model.to(device)
            if hasattr(benchmark_model, 'ax') and args.use_bf16:
                benchmark_model.ax.eval()
                benchmark_model.ax.to(torch.bfloat16)
            # we only save the max act for each concept to save disk space, otherwise each file will be ~3GB.
            # if you wish to save the raw acts, you can go into predict_latents and modify the output.
            results = benchmark_model.predict_latents(
                all_negative_df, 
                batch_size=args.latent_batch_size, 
                prefix_length=prefix_length
            )
            # save results to disk
            with open(dump_dir / f"{model_name}_latent_results.pkl", "wb") as f:
                pickle.dump(results, f)


def infer_latent_on_train_data(args, rank, world_size, device, logger, training_args, generate_args):
    """This is used for getting threshold for latent and steering."""
    data_dir = args.data_dir
    train_dir = args.train_dir
    dump_dir = args.dump_dir
    num_of_examples = args.latent_num_of_examples
    config = load_config(train_dir)
    metadata = load_metadata_flatten(data_dir)
    layer = config["layer"] if config else 0  # default layer for prompt baselines

    state = load_state(args.dump_dir, "latent_on_train_data", rank)
    last_concept_id_processed = state.get("last_concept_id", None) if state else None
    logger.warning(f"Rank {rank} last concept_id processed: {last_concept_id_processed}")

    # Get list of all concept_ids
    concept_ids = [metadata[i]["concept_id"] for i in range(len(metadata))]

    # Partition concept_ids among ranks sequentially
    assert world_size == 1, "latent_on_train_data only supports world_size = 1"
    concept_ids_per_rank = partition_concept_ids(concept_ids, world_size)
    my_concept_ids = concept_ids_per_rank[rank]

    if last_concept_id_processed is not None:
        if last_concept_id_processed in my_concept_ids:
            idx = my_concept_ids.index(last_concept_id_processed)
            my_concept_ids = my_concept_ids[idx+1:]
        else:
            # If last_concept_id_processed is not in my_concept_ids, process all
            pass

    if len(my_concept_ids) == 0:
        logger.warning(f"Rank {rank} has no concepts to process. Exiting.")
        return

    # Create a new OpenAI client.
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=60.0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=100,
                max_connections=1000
            ),
            headers={"Connection": "close"},
        ),
        max_retries=3,
    )

    dump_dir = Path(dump_dir) / "inference_on_train_data"
    dump_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, model_max_length=1024)
    tokenizer.padding_side = "right"

    # Load model instance onto device
    if args.use_bf16:
        logger.warning(f"Using bfloat16 for model {args.model_name}")
    model_instance = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        torch_dtype=torch.bfloat16 if args.use_bf16 else None, 
        device_map=device
    )
    is_chat_model = True if args.model_name in CHAT_MODELS else False
    model_instance = model_instance.eval()

    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        need_resize = True
    else:
        need_resize = False
    if need_resize:
        model_instance.resize_token_embeddings(len(tokenizer))

    prefix_length = 1 # prefix is default to 1 for all models due to the BOS token.
    if is_chat_model:
        prefix_length = get_prefix_length(tokenizer)
        logger.warning(f"Chat model prefix length: {prefix_length}")

    # Load dataset factory for evals.
    dataset_factory = DatasetFactory(
        None, client, tokenizer, generate_args.dataset_category, None, None, dump_dir,
        use_cache=False, master_data_dir=args.master_data_dir,
        lm_model=args.lm_model, logger=logger, is_inference=True,
        overwrite_inference_data_dir=training_args.overwrite_inference_data_dir
    )
    atexit.register(dataset_factory.save_cache)
    atexit.register(dataset_factory.reset_stats)

    has_latent_model = False
    for model_name in args.models:
        # load model on the fly to save memory
        if model_name not in LATENT_EXCLUDE_MODELS:
            has_latent_model = True
            break

    if not has_latent_model:
        logger.warning("No latent model to infer. Exiting.")
        return

    # Now loop over concept_ids and use preloaded models
    cache_df = {}
    all_results = {}
    for model_name in args.models:
        all_results[model_name] = {}
    concept_count = 0
    for concept_id in my_concept_ids:
        current_df = create_data_latent(
            dataset_factory, metadata, concept_id, num_of_examples, args)
        current_df = prepare_df(current_df, tokenizer, is_chat_model, args.model_name)
        if len(current_df) == 0:
            # for cases where the concept_id is not in the dataset, we skip it.
            # we dont increment concept_count in this case.
            continue
        for model_name in args.models:
            logger.warning(f"Inference latent with {model_name} on {device} for concept {concept_id}.")
            # load model on the fly to save memory
            if model_name in LATENT_EXCLUDE_MODELS:
                continue
            model_class = getattr(axbench, model_name)
            logger.warning(f"Loading {model_class} on {device}.")
            benchmark_model = model_class(
                model_instance, tokenizer, layer=layer,
                low_rank_dimension=len(metadata),
                device=device
            )
            benchmark_model.load(
                dump_dir=train_dir, sae_path=metadata[0]["ref"], mode="latent"
            )
            benchmark_model.to(device)
            if hasattr(benchmark_model, 'ax') and args.use_bf16:
                benchmark_model.ax.eval()
                benchmark_model.ax.to(torch.bfloat16)

            results = benchmark_model.predict_latent(
                current_df, batch_size=args.latent_batch_size, prefix_length=prefix_length, 
                return_max_act_only=True, overwrite_concept_id=concept_count
            )
            all_results[model_name][concept_id] = results
            del benchmark_model
            torch.cuda.empty_cache()
        concept_count += 1
        if concept_count % 500 == 0 or concept_id == my_concept_ids[-1]:
            rotation_index = (concept_count-1) // 500
            # save results to disk
            with open(dump_dir / f"rank_{rank}_all_results_{rotation_index}.pkl", "wb") as f:
                pickle.dump(all_results, f)
            # clear all_results
            all_results = {}
            for model_name in args.models:
                all_results[model_name] = {}

    # Synchronize all processes
    dist.barrier()

    # Rank 0 merges results
    if rank == 0:
        logger.warning("All ranks have finished inference.")


def _inference_dir(args):
    if getattr(args, "overwrite_inference_dump_dir", None) is not None:
        return Path(args.overwrite_inference_dump_dir)
    return Path(args.dump_dir) / "inference"


def _chunk_set_complete(args, mode):
    """
    True once every concept expected for this dataset has a row in some
    rank_*_{mode}_chunk*_data.parquet file already on disk -- i.e. every
    --chunk job in the --num_chunks sweep has finished writing its output.
    """
    inference_dir = _inference_dir(args)
    paths = glob.glob(str(inference_dir / f"rank_*_{mode}_chunk*_data.parquet"))
    if not paths:
        return False
    expected = {m["concept_id"] for m in load_metadata_flatten(args.data_dir)}
    found = set()
    for p in paths:
        found.update(pd.read_parquet(p, columns=["concept_id"])["concept_id"].unique())
    return expected.issubset(found)


def _chunk_already_merged(args, mode, window_ids):
    """True if the merged {mode}_data.parquet already covers every concept in window_ids."""
    path = _inference_dir(args) / f"{mode}_data.parquet"
    if not path.exists():
        return False
    merged = set(pd.read_parquet(path, columns=["concept_id"])["concept_id"].unique())
    return set(window_ids).issubset(merged)


def maybe_auto_merge_chunks(args, mode, logger):
    """
    Called by a --chunk job right after it finishes its own slice: if every
    concept is now covered across all rank_*_{mode}_chunk*_data.parquet files
    (i.e. this was the last chunk in the --num_chunks sweep to finish), merge
    them into {mode}_data.parquet immediately -- a separate --merge_chunks
    invocation after the sweep is no longer required. Once the merge passes
    verification, the chunk files and their resume state are deleted.

    If a sibling chunk job's own auto-merge races this one, the loser either
    fails reading a chunk file the winner already deleted, or merges a partial
    set that fails verification -- either way it logs and leaves the winner's
    merged file untouched.
    """
    try:
        if _chunk_set_complete(args, mode):
            logger.warning(f"All chunks present for {mode} -- auto-merging.")
            merge_chunks(args, mode, logger, cleanup=True)
    except Exception as e:
        logger.warning(f"Auto-merge skipped ({e}); run --merge_chunks manually if needed.")


def merge_chunks(args, mode, logger, cleanup=False):
    """
    Combine the per-chunk parquets written by N independent `--chunk` jobs into the
    single unchunked {mode}_data.parquet the downstream stages expect.

    Only latent's merged filename matches the `latent_*.parquet` glob that
    Model.pre_compute_mean_activations scans, so steering cannot read chunk files
    directly -- latent must be merged before any steering job starts.

    The merge is written to a temporary file and checked (_check_merged) before it
    replaces {mode}_data.parquet, so a failed or partial merge never overwrites a good
    one. With cleanup=True, the merged chunk files and every chunk's resume state are
    then deleted; otherwise they are left in place.
    """
    inference_dir = _inference_dir(args)
    paths = sorted(
        glob.glob(str(inference_dir / f"rank_*_{mode}_chunk*_data.parquet")),
        key=lambda p: int(re.search(r"_chunk(\d+)_data\.parquet$", p).group(1)))
    if not paths:
        raise FileNotFoundError(
            f"No chunk files matching rank_*_{mode}_chunk*_data.parquet in {inference_dir}")
    logger.warning(f"Merging {len(paths)} chunk file(s): {[Path(p).name for p in paths]}")

    # A concept legitimately spans many rows within one chunk file, so overlap has to be
    # checked per source file rather than by counting rows in the concatenation.
    seen = {}
    for p in paths:
        for cid in pd.read_parquet(p, columns=["concept_id"])["concept_id"].unique():
            seen.setdefault(cid, []).append(Path(p).name)
    dupes = {cid: files for cid, files in seen.items() if len(files) > 1}
    if dupes:
        sample = list(dupes.items())[:5]
        raise ValueError(
            f"{len(dupes)} concept_id(s) appear in more than one chunk file -- refusing to "
            f"merge. Chunk windows overlap, or stale files from a run with a different "
            f"--num_chunks are present in {inference_dir}. Examples: {sample}")

    combined = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    counts = combined.groupby("concept_id").size()

    if mode == "steering":
        combined = combined.sort_values(
            by=["concept_id", "input_id", "factor"]).reset_index(drop=True)
    else:
        combined = combined.sort_values(by=["concept_id"]).reset_index(drop=True)

    out = inference_dir / f"{mode}_data.parquet"
    if not _check_merged(args, mode, combined, f"merge of {len(paths)} chunk file(s)", logger):
        raise ValueError(
            f"Merged {mode} data failed verification -- {out} left untouched and chunk "
            f"files left in place.")
    # Unique temp name, so two racing mergers never write the same file.
    tmp = inference_dir / f"{mode}_data.parquet.tmp.{uuid.uuid4().hex}"
    try:
        combined.to_parquet(tmp, engine="pyarrow")
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()
    logger.warning(f"Wrote {out} -- {len(combined)} rows across {counts.shape[0]} concepts.")

    if cleanup:
        stale = paths + glob.glob(str(inference_dir / f"{mode}_chunk*_{STATE_FILE}_rank_*"))
        for p in stale:
            Path(p).unlink(missing_ok=True)
        logger.warning(f"Deleted {len(stale)} merged chunk/state file(s).")
        if mode == "steering":
            consolidate_steering_eval_cache(
                args.dump_dir, logger, getattr(args, "steering_instructions_dist", "train"))
    else:
        logger.warning("Per-chunk files left in place; delete them once you are satisfied.")


def _check_merged(args, mode, df, label, logger):
    """
    True if df is a complete, usable merged {mode} parquet.

    Checks concept coverage against the metadata, and for latent that every model has
    a populated {ModelName}_max_act -- the column steering multiplies its factors by.
    A missing or all-null column there is the silent failure where every steering
    factor quietly falls back to 1.0.
    """
    expected = sorted(m["concept_id"] for m in load_metadata_flatten(args.data_dir))
    found = sorted(df["concept_id"].unique())
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))

    ok = True
    logger.warning(f"{label}: {len(df)} rows, {len(found)}/{len(expected)} concepts")
    if missing:
        ok = False
        logger.warning(f"  MISSING {len(missing)} concept_id(s), first 10: {missing[:10]}")
    if extra:
        ok = False
        logger.warning(f"  UNEXPECTED concept_id(s), first 10: {extra[:10]}")

    if mode == "latent":
        for model_name in getattr(args, "models", []) or []:
            col = f"{model_name}_max_act"
            if col not in df.columns:
                ok = False
                logger.warning(f"  MISSING column {col} -- steering would fall back to 1.0")
                continue
            per_concept = df.groupby("concept_id")[col].max()
            n_null = int(per_concept.isna().sum())
            n_nonpos = int((per_concept <= 0).sum())
            if n_null:
                ok = False
            logger.warning(
                f"  {col}: mean={per_concept.mean():.2f} min={per_concept.min():.2f} "
                f"null={n_null} non-positive={n_nonpos}"
                + (f"  <- these {n_nonpos} fall back to 50" if n_nonpos else ""))

    return ok


def verify_chunks(args, mode, logger):
    """Gate the next pipeline stage on a complete, usable merged parquet."""
    path = _inference_dir(args) / f"{mode}_data.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist -- run --merge_chunks first.")
    if not _check_merged(args, mode, pd.read_parquet(path), path.name, logger):
        raise SystemExit(f"verify_chunks FAILED for {path} -- do not start the next stage.")
    logger.warning(f"verify_chunks PASSED for {path}")


def main():
    start_time = time.time()
    custom_args = [
        {
            'args': ['--mode'],
            'kwargs': {
                'type': str,
                'default': "all",
                'help': 'The inference mode.'
            }
        },
        {
            'args': ['--chunk'],
            'kwargs': {
                'type': int,
                'default': None,
                'help': '0-based index of the concept chunk to process. Selects one window '
                        'of the --num_chunks-way split of the concept set. Every output '
                        'file is tagged so concurrent chunks never collide.'
            }
        },
        {
            'args': ['--num_chunks'],
            'kwargs': {
                'type': int,
                'default': 5,
                'help': 'Total number of chunks the concept set is split into. Concepts are '
                        'divided into this many contiguous windows; --chunk selects one.'
            }
        },
        {
            'args': ['--merge_chunks'],
            'kwargs': {
                'action': 'store_true',
                'help': 'Merge per-chunk parquets for --mode into {mode}_data.parquet, then exit. '
                        'The merge is verified before it is written; with --verify_chunks '
                        'also passed, the merged chunk files are then deleted.'
            }
        },
        {
            'args': ['--verify_chunks'],
            'kwargs': {
                'action': 'store_true',
                'help': 'Check the merged {mode}_data.parquet for full concept coverage (and, '
                        'for latent, populated max_act columns), then exit.'
            }
        }
    ]
    training_args = TrainingArgs(custom_args=custom_args, section="train", ignore_unknown=True)
    generate_args = DatasetArgs(custom_args=custom_args, section="generate", ignore_unknown=True)
    inference_args = DatasetArgs(custom_args=custom_args, section="inference", ignore_unknown=True)

    if training_args.overwrite_metadata_dir is not None and os.path.exists(training_args.overwrite_metadata_dir):
        inference_args.data_dir = training_args.overwrite_metadata_dir # since we only load metadata from this dir
    else:
        inference_args.data_dir = f"{inference_args.dump_dir}/generate"
    inference_args.train_dir = f"{inference_args.dump_dir}/train"
    # The args classes only copy attributes they recognise from the YAML section, so a
    # flag that is merely absent from the command line never gets set at all.
    for _name, _default in (
            ("chunk", None), ("num_chunks", 5),
            ("merge_chunks", False), ("verify_chunks", False)):
        if not hasattr(inference_args, _name):
            setattr(inference_args, _name, _default)

    logger.warning("Inferencing with following configuration:")
    logger.warning(inference_args)
    set_seed(inference_args.seed)

    # Merging and verifying are CPU-only bookkeeping over parquets. Handle them before
    # the process group is created so they can run as a plain `uv run`, without torchrun
    # and without holding a GPU.
    if inference_args.merge_chunks or inference_args.verify_chunks:
        if inference_args.merge_chunks:
            # Deleting the chunk files is opt-in here: only when --verify_chunks is
            # passed too. The merge itself is always verified before it is written.
            merge_chunks(inference_args, inference_args.mode, logger,
                         cleanup=inference_args.verify_chunks)
        if inference_args.verify_chunks:
            verify_chunks(inference_args, inference_args.mode, logger)
        elapsed = time.time() - start_time
        logger.warning(f"Total time taken: {elapsed:.1f}s ({datetime.timedelta(seconds=int(elapsed))})")
        return

    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://',
                          timeout=datetime.timedelta(seconds=60000))


    # Get the rank and world_size from environment variables
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # Set the device for this process
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    # Configure the logger per rank
    logger.setLevel(logging.WARNING)  # Set the logging level as desired

    # Create a logging formatter that includes the rank
    formatter = logging.Formatter(
        fmt=f'%(asctime)s,%(msecs)03d %(levelname)-8s [Rank {rank}] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S'
    )

    # Create a console handler and set its formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    if not logger.handlers:
        logger.addHandler(console_handler)

    # Optionally, create a file handler per rank
    """
    log_file = f'log_rank_{rank}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    """

    # Add suppress_eval_dir to inference_args if present in command line
    if hasattr(inference_args, 'suppress_eval_dir'):
        suppress_eval_dir = inference_args.suppress_eval_dir
    else:
        suppress_eval_dir = None

    if inference_args.mode == "latent":
        infer_latent(inference_args, rank, world_size, device, logger, training_args, generate_args)
    elif inference_args.mode == "latent_imbalance":
        infer_latent_imbalance(inference_args, rank, world_size, device, logger, training_args, generate_args)
    elif inference_args.mode == "latent_on_train_data":
        infer_latent_on_train_data(inference_args, rank, world_size, device, logger, training_args, generate_args)
    elif inference_args.mode == "steering":
        infer_steering(inference_args, rank, world_size, device, logger, training_args, generate_args, suppress_eval_dir=suppress_eval_dir)
    elif inference_args.mode == "all":
        infer_latent(inference_args, rank, world_size, device, logger, training_args, generate_args)
        infer_steering(inference_args, rank, world_size, device, logger, training_args, generate_args, suppress_eval_dir=suppress_eval_dir)

    if rank == 0:
        elapsed = time.time() - start_time
        logger.warning(f"Total time taken by rank 0: {elapsed:.1f}s ({datetime.timedelta(seconds=int(elapsed))})")

    # Finalize the process group
    dist.destroy_process_group()

    # Remove handlers to prevent duplication if the script is run multiple times
    logger.removeHandler(console_handler)
    # If file_handler is used, remove it as well
    # logger.removeHandler(file_handler)


if __name__ == "__main__":
    main()

