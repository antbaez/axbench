# generate script for creating the training dataset for concepts.
# we assume we group two concepts into a learning group.
# it is possible to extend to more concepts into the same group,
# although more training data will likely to be needed.
# 
# example launch command:
#    python axbench/scripts/generate.py --config axbench/demo/sweep/generate.yaml

import warnings
warnings.filterwarnings("ignore", message=r"pyreft not installed.*")
warnings.filterwarnings("ignore", message=r"HyperSteer unavailable.*")

import shutil
import sys
import argparse
import time
import os
import pickle
import random
import json
import csv
import atexit
import threading
import torch
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from axbench.utils.dataset import (
    CONTRASTIVE_CONCEPTS_FILE,
    DatasetFactory,
    load_base_instructions,
    load_or_create_base_instructions,
    load_contrastive_concepts,
    load_or_create_contrastive_concepts_batch,
)
from axbench.models.language_models import RequestLimiter
from args.dataset_args import DatasetArgs
from pathlib import Path
from openai import AsyncOpenAI
import httpx, asyncio
from transformers import set_seed
from axbench.utils.constants import * 

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)

model_name_map = {
    "gemma-2-2b": "google/gemma-2-2b-it",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}

MAX_RETRIES = 5
RETRY_DELAY = 1  # in seconds
# extra 429 retries (exponential backoff, capped at ~60s) beyond the OpenAI client's own
RATE_LIMIT_RETRIES = 8
STATE_FILE = "generate_state.pkl"
METADATA_FILE = "metadata.jsonl"
PREVIEW_FILE = "train_data_preview.json"
PREVIEW_CONCEPTS = 10
PREVIEW_PAIRS = 10
SELECTED_CONCEPTS_FILE = "selected_concepts.json"
# concept names classified per genre LLM call while filling up to max_concepts
GENRE_BATCH_SIZE = 256


def make_openai_client():
    return AsyncOpenAI(
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


def load_concepts(dump_dir):
    sae_concepts = []
    if ".txt" in dump_dir:
        with open(dump_dir, 'r') as file:
            concepts = [line.strip() for line in file.readlines()]
        if concepts[0].startswith("http://") or concepts[0].startswith("https://"):
            logger.warning("Detect external links. Pull concept info from the link.")
            for concept in concepts:
                if "www.neuronpedia.org" not in concept:
                    raise ValueError(f"Pulling from {concept} is not supported.")
                sae_path = concept.split("https://www.neuronpedia.org/")[-1]
                sae_url = f"https://www.neuronpedia.org/api/feature/{sae_path}"
                headers = {"X-Api-Key": os.environ.get("NP_API_KEY")}
                response = requests.get(sae_url, headers=headers).json()
                explanation = response["explanations"][0]["description"]
                sae_concepts += [explanation.strip()]
            return sae_concepts, concepts
        return concepts, ["null"]*len(concepts)
    elif ".csv" in dump_dir:
        # for csv, then the format is <concept>,<url>
        # no http connection is needed
        concepts = []
        with open(dump_dir, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                sae_concepts += [row[0]]
                concepts += [row[1]]
        return sae_concepts, concepts
    elif ".json" in dump_dir:
        concepts = []
        # this must be a neuropedia export.
        with open(dump_dir, 'r') as file:
            json_concepts = json.load(file)
        seen_index = set()
        for concept in json_concepts:
            model = concept["modelId"]
            sae_model = concept["layer"]
            subspace_id = concept["index"]
            if subspace_id in seen_index:
                continue # if there are multiple descriptions, we only take the first one.
            seen_index.add(subspace_id)
            sae_concepts += [concept["description"].strip()]
            concepts += [f"https://www.neuronpedia.org/{model}/{sae_model}/{subspace_id}"]
        return sae_concepts, concepts
    else:
        raise ValueError(f"Unsupported file type: {dump_dir}.")  


def save_df_to_parquet_safely(df, final_path):
    import tempfile
    import os
    
    # Create temporary file in the same directory as the target
    dirname = os.path.dirname(os.path.abspath(final_path))
    with tempfile.NamedTemporaryFile(delete=False, dir=dirname, suffix='.parquet.tmp') as tmp:
        temp_path = tmp.name
        try:
            # Write to temporary file first
            df.to_parquet(temp_path, index=False)
            # Ensure data is written to disk
            os.fsync(tmp.fileno())
        except Exception as e:
            os.unlink(temp_path)  # Clean up temp file
            raise e
    
    try:
        # Atomic rename operation
        os.rename(temp_path, final_path)
    except Exception as e:
        os.unlink(temp_path)  # Clean up temp file
        raise e
    

def load_metadata_flatten(metadata_path):
    """
    Load flatten metadata from a JSON lines file.
    """
    metadata = []
    with open(Path(metadata_path) / METADATA_FILE, 'r') as f:
        for line in f:
            data = json.loads(line)
            concept, ref =data["concept"], data["ref"]
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
    dump_dir, state, concept_id, 
    concept, concept_genres_map, 
    ref, partition, current_df, dataset_factory):
    """
    Save the current state, metadata, and DataFrame using Parquet format.
    """    
    # Save state
    state_path = os.path.join(dump_dir, STATE_FILE)
    with open(state_path, "wb") as f:
        pickle.dump(state, f)
    
    # Save metadata
    metadata_path = os.path.join(dump_dir, METADATA_FILE)
    metadata_entry = {
        "concept_id": concept_id,
        "concept": concept,
        "ref": ref,
        "concept_genres_map": concept_genres_map,
    }
    with open(metadata_path, "a") as f:
        f.write(json.dumps(metadata_entry) + "\n")
    
    # Save DataFrame using Parquet
    rotation_freq = 500
    file_index = concept_id // rotation_freq
    if file_index == 0:
        df_path = os.path.join(dump_dir, f"{partition}_data.parquet")
    else:
        df_path = os.path.join(dump_dir, f"{partition}_data_{file_index}.parquet")
    if os.path.exists(df_path):
        existing_df = pd.read_parquet(df_path)
        combined_df = pd.concat([existing_df, current_df], ignore_index=True)
    else:
        # first time cache, we need to add global negative examples.
        if concept_id == 0:
            combined_df = pd.concat([dataset_factory.negative_df, current_df], ignore_index=True)
        else:
            combined_df = current_df
    save_df_to_parquet_safely(combined_df, df_path)

    # Also save the same data as JSON for easy human inspection.
    json_path = os.path.splitext(df_path)[0] + ".json"
    combined_df.to_json(json_path, orient="records", indent=2, force_ascii=False)


def save_preview(dump_dir, partition):
    """First PREVIEW_PAIRS positive/negative pairs of the first PREVIEW_CONCEPTS concepts.

    A small file for eyeballing data quality early in a run, instead of the full
    {partition}_data.json. Built from the saved parquet, so it comes out the same
    whether or not the run was resumed along the way.
    """
    # concept ids below the parquet rotation size all live in the first file
    df = pd.read_parquet(os.path.join(dump_dir, f"{partition}_data.parquet"))
    df = df[df["concept_id"] >= 0]
    preview = []
    for concept_id in sorted(df["concept_id"].unique())[:PREVIEW_CONCEPTS]:
        rows = df[df["concept_id"] == concept_id]
        positives = rows[rows["category"] == "positive"]
        negatives = rows[rows["category"] == "negative"]
        # create_train_df emits each negative in the same order as the positive it edits
        pairs = [
            {
                "base_instruction": base_instruction, "positive": positive,
                "negative": negative, "contrast_concept": contrast_concept,
            }
            for base_instruction, positive, negative, contrast_concept in zip(
                positives["base_instruction"], positives["input"], negatives["input"],
                negatives["contrast_concept"])
        ][:PREVIEW_PAIRS]
        preview.append({
            "concept_id": int(concept_id),
            "concept": positives["output_concept"].iloc[0],
            "pairs": pairs,
        })
    with open(os.path.join(dump_dir, PREVIEW_FILE), "w") as f:
        json.dump(preview, f, indent=2, ensure_ascii=False)
    logger.warning(f"Wrote {PREVIEW_FILE} with {len(preview)} concept(s).")


def load_state(dump_dir):
    """
    Load the state from a file if it exists.
    
    Args:
        dump_dir (str): The directory to load the state file from.
    
    Returns:
        dict: The loaded state dictionary, or None if no state file exists.
    """
    state_path = os.path.join(Path(dump_dir), STATE_FILE)
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            state = pickle.load(f)
            return state
    return None


def count_metadata_entries(dump_dir):
    """How many concepts have already been written, i.e. the next data concept_id.

    One metadata line is appended per concept that produced data, so this stays correct
    across resumes even when concepts in between were skipped by the genre filter.
    """
    metadata_path = os.path.join(Path(dump_dir), METADATA_FILE)
    if not os.path.exists(metadata_path):
        return 0
    with open(metadata_path) as f:
        return sum(1 for line in f if line.strip())


def sae_index(ref):
    """Feature index from a neuronpedia ref (.../<model>/<sae>/<index>), or None."""
    try:
        return int(str(ref).rstrip("/").split("/")[-1])
    except ValueError:
        return None


def select_text_concepts(all_concepts, all_refs, max_concepts, seed,
                         seed_concepts_dir, dataset_factory, dump_dir,
                         concept16k_dir=None):
    """
    Pick the text-genre concepts to generate for: up to max_concepts of them.

    Text concepts from seed_concepts_dir (a prior run's generate/ dir, e.g. concept500)
    come first, in their original order and with the genres that run recorded. Every
    seed concept is excluded from every later fill whatever its genre, so the seed
    run's verdict stands -- re-asking the genre model could flip a code concept to text.

    If more are still needed, concept16k_dir (a concept16k split's generate/ dir) fills
    the shortfall next: its concepts are already genre-labeled by the same pipeline, so
    no LLM call is needed, just a filter to genre "text" and exclusion of anything
    already taken from the seed. The sample is drawn with a fixed seed (42), independent
    of this run's own seed, so the concept16k fill is reproducible across runs.

    Only if concept16k_dir is absent, or still can't cover the shortfall, does the old
    fallback run: a seeded shuffle of the full concept pool, genre-classified via LLM in
    batches until max_concepts text concepts are found.

    Genre labels for the LLM-classified fallback come from a temperature-1.0 call, so
    the choice is written to SELECTED_CONCEPTS_FILE once and reloaded on every later
    run; recomputing it on a resume could hand back a different concept list than the
    one already half-generated.
    """
    path = os.path.join(dump_dir, SELECTED_CONCEPTS_FILE)
    if os.path.exists(path):
        with open(path) as f:
            selected = json.load(f)["concepts"]
        logger.warning(f"Loaded {len(selected)} concepts from {path}.")
        return selected

    selected, taken, taken_refs = [], set(), set()
    if seed_concepts_dir:
        for row in load_metadata_flatten(seed_concepts_dir):
            concept, genres = row["concept"], row["concept_genres_map"][row["concept"]]
            taken.add(concept)
            taken_refs.add(row["ref"])
            if genres[0] == "text" and (max_concepts is None or len(selected) < max_concepts):
                selected.append({"concept": concept, "ref": row["ref"],
                                 "genres": genres, "source": "seed"})
        logger.warning(
            f"{len(selected)} text concepts taken from {seed_concepts_dir} "
            f"({len(taken)} seed concepts excluded from the random fill).")

    if concept16k_dir and (max_concepts is None or len(selected) < max_concepts):
        n_before = len(selected)
        candidates = []
        for row in load_metadata_flatten(concept16k_dir):
            concept, genres = row["concept"], row["concept_genres_map"][row["concept"]]
            if concept in taken or row["ref"] in taken_refs:
                continue
            if genres[0] != "text":
                continue
            candidates.append((concept, row["ref"], genres))
        # Fixed seed (42), independent of this run's own seed, so the concept16k fill
        # is reproducible across runs regardless of --seed.
        random.Random(42).shuffle(candidates)
        n_needed = None if max_concepts is None else max_concepts - len(selected)
        for concept, ref, genres in candidates:
            if n_needed is not None and len(selected) - n_before >= n_needed:
                break
            taken.add(concept)
            taken_refs.add(ref)
            selected.append({"concept": concept, "ref": ref,
                             "genres": genres, "source": "concept16k"})
        logger.warning(
            f"{len(selected) - n_before} text concepts taken from {concept16k_dir} "
            f"({len(candidates)} eligible after excluding seed concepts).")

    # Same seeded shuffle of the full pool that the old slice-then-filter code used, so
    # with a concept500 seed the fill continues where concept500's own sample stopped.
    pool = list(zip(all_concepts, all_refs))
    random.Random(seed).shuffle(pool)
    pool = [(c, r) for c, r in pool if c not in taken and r not in taken_refs]

    n_classified, i = 0, 0
    while (max_concepts is None or len(selected) < max_concepts) and i < len(pool):
        # distinct features can share a description, and every downstream map is keyed
        # by the description, so a repeat would collide with the concept already taken
        batch = []
        while len(batch) < GENRE_BATCH_SIZE and i < len(pool):
            concept, ref = pool[i]
            i += 1
            if concept not in taken:
                taken.add(concept)
                batch.append((concept, ref))
        if not batch:
            break
        genres_map = dataset_factory.prepare_genre_concepts([c for c, _ in batch])
        n_classified += len(batch)
        for concept, ref in batch:
            if genres_map[concept][0] != "text":
                continue
            selected.append({"concept": concept, "ref": ref,
                             "genres": genres_map[concept], "source": "random"})
            if max_concepts is not None and len(selected) >= max_concepts:
                break
    if max_concepts is not None and len(selected) < max_concepts:
        logger.warning(
            f"Only {len(selected)} text concepts available, fewer than max_concepts="
            f"{max_concepts}; continuing with those.")

    for concept_id, entry in enumerate(selected):
        entry["concept_id"] = concept_id
        entry["sae_index"] = sae_index(entry["ref"])
    n_seed = sum(e["source"] == "seed" for e in selected)
    n_concept16k = sum(e["source"] == "concept16k" for e in selected)
    n_random = len(selected) - n_seed - n_concept16k
    with open(path, "w") as f:
        json.dump({
            "max_concepts": max_concepts,
            "seed": seed,
            "seed_concepts_dir": seed_concepts_dir,
            "concept16k_dir": concept16k_dir,
            "n_from_seed": n_seed,
            "n_from_concept16k": n_concept16k,
            "n_random": n_random,
            "n_genre_classified": n_classified,
            "concepts": selected,
        }, f, indent=2, ensure_ascii=False)
    logger.warning(
        f"Selected {len(selected)} text concepts ({n_seed} from seed, {n_concept16k} from "
        f"concept16k, {n_random} random after classifying {n_classified}); wrote {path}.")
    return selected


def load_state_latent(dump_dir, mode):
    """
    Load the state from a file if it exists.
    """
    state_path = os.path.join(f"{dump_dir}/generate", f"{mode}_{STATE_FILE}")
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            return pickle.load(f)
    return None


def create_data_latent(
        dataset_factory, metadata, concept_id, num_of_examples, args,
        base_instructions, contrastive_concepts_store, lm_model=None, rng=None):
    # prepare concept related data.
    concept = metadata[concept_id]["concept"]
    sae_link = metadata[concept_id]["ref"]
    try:
        sae_id = int(sae_link.split("/")[-1])
    except:
        sae_id = 0
    concept_genres_map = metadata[concept_id]["concept_genres_map"]
    rng = rng or random
    # Same builder as --mode training, drawing from the same shared stores: a random
    # subset of the base instructions, and this concept's own stored contrast pool.
    sampled_instructions = rng.sample(
        base_instructions, min(num_of_examples, len(base_instructions)))
    current_df = dataset_factory.create_train_df(
        concept, concept_genres_map, sampled_instructions,
        contrastive_concepts_store[concept],
        output_length=int(args.output_length),
        lm_model=lm_model, rng=rng)
    current_df["concept_id"] = concept_id
    current_df["sae_link"] = sae_link
    current_df["sae_id"] = sae_id
    return current_df


def save_state_latent(dump_dir, state, partition):
    dump_dir = Path(dump_dir) / "generate"
    dump_dir.mkdir(parents=True, exist_ok=True)
    # Save state
    state_path = os.path.join(dump_dir, f"{partition}_{STATE_FILE}")
    with open(state_path, "wb") as f:
        pickle.dump(state, f)


def save_latent(
    dump_dir, concept_id, partition,
    current_df):
    # This function saves DataFrames per rank per partition (latent or steering)
    dump_dir = Path(dump_dir) / "generate"
    dump_dir.mkdir(parents=True, exist_ok=True)
    
    # Save DataFrame using Parquet
    rotation_freq = 500
    file_index = concept_id // rotation_freq
    if file_index == 0:
        df_path = os.path.join(dump_dir, f"{partition}_eval_data.parquet")
    else:
        df_path = os.path.join(dump_dir, f"{partition}_eval_data_{file_index}.parquet")
    if os.path.exists(df_path):
        existing_df = pd.read_parquet(df_path)
        combined_df = pd.concat([existing_df, current_df], ignore_index=True)
    else:
        combined_df = current_df
    # atomic, so a preemption mid-write can't corrupt every concept saved so far
    save_df_to_parquet_safely(combined_df, df_path)


def generate_latent(generate_args, args):
    args.data_dir = f"{args.dump_dir}/generate"
    logger.warning("Inferencing with following configuration:")
    logger.warning(args)
    set_seed(args.seed)

    # Configure the logger per rank
    logger.setLevel(logging.WARNING)  # Set the logging level as desired

    # Create a logging formatter that includes the rank
    formatter = logging.Formatter(
        fmt=f'%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
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
    data_dir = args.data_dir
    dump_dir = args.dump_dir
    num_of_examples = args.latent_num_of_examples
    metadata = load_metadata_flatten(data_dir)
    # Get list of all concept_ids
    concept_ids = list(range(len(metadata)))

    # Load the state if it exists.
    state = load_state_latent(args.dump_dir, "latent")
    start_concept_id = state.get("concept_id", 0) if state else 0
    logger.warning(f"Starting concept index: {start_concept_id}")
    if start_concept_id >= len(concept_ids):
        logger.warning(f"Datasets for all concepts have been generated. Exiting.")
        return

    # Create a new OpenAI client (worker threads each make their own, see below).
    # Concurrency settings come from the generate: section, as in --mode training.
    client = make_openai_client()
    num_workers = int(generate_args.num_workers or 1)
    request_limiter = RequestLimiter(int(generate_args.max_concurrent_requests)) \
        if generate_args.max_concurrent_requests else None
    logger.warning(
        f"Generating with {num_workers} worker(s), max concurrent requests: "
        f"{generate_args.max_concurrent_requests or f'uncapped (up to {64 * num_workers})'}.")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, model_max_length=512)
    tokenizer.padding_side = "right"

    # Load dataset factory for evals.
    dataset_factory = DatasetFactory(
        None, client, tokenizer, generate_args.dataset_category, None, None, dump_dir,
        use_cache=args.lm_use_cache, master_data_dir=args.master_data_dir,
        lm_model=args.lm_model, logger=logger, is_inference=True,
        request_limiter=request_limiter, rate_limit_retries=RATE_LIMIT_RETRIES,
    )
    atexit.register(dataset_factory.save_cache)
    atexit.register(dataset_factory.reset_stats)

    # Both stores are written by --mode training; latent only reads them, so its
    # prompts are built from the same instructions and contrast concepts as training's.
    base_instructions = load_base_instructions(data_dir)
    contrastive_concepts_store = load_contrastive_concepts(data_dir)

    remaining_ids = range(start_concept_id, len(metadata))
    # checked up front so a run that would fail does so before spending anything
    missing = [metadata[i]["concept"] for i in remaining_ids
               if metadata[i]["concept"] not in contrastive_concepts_store]
    if missing:
        raise KeyError(
            f"{len(missing)} concept(s) missing from {CONTRASTIVE_CONCEPTS_FILE} "
            f"(e.g. '{missing[0]}') -- it is written by --mode training, which must run first.")

    thread_state = threading.local()

    def generate_concept(idx):
        # an httpx client is tied to the event loop it runs on, so each thread needs
        # its own; stats, cache and request limiter stay shared.
        if not hasattr(thread_state, "lm_model"):
            thread_state.lm_model = dataset_factory.lm_model.fork(make_openai_client())
        concept_id = metadata[idx]["concept_id"]
        return create_data_latent(
            dataset_factory, metadata, concept_id, num_of_examples, args,
            base_instructions, contrastive_concepts_store,
            lm_model=thread_state.lm_model,
            # seeded by concept rather than drawn from the global RNG, so the sampled
            # instructions don't depend on thread timing or where a resume started
            rng=random.Random(f"{args.seed}:latent:{concept_id}"))

    executor = ThreadPoolExecutor(max_workers=num_workers)
    futures = {idx: executor.submit(generate_concept, idx) for idx in remaining_ids}
    try:
        # Workers finish in any order; results are saved strictly in concept order,
        # which the appended parquet and the single resume watermark rely on.
        for idx in tqdm(remaining_ids, desc="Processing concept"):
            concept_id = metadata[idx]["concept_id"]
            current_df = futures.pop(idx).result()

            save_latent(dump_dir, concept_id, 'latent', current_df)
            logger.info(f"Saved inference dataset for concept {concept_id} to latent_eval_data.parquet")
            # The next concept to generate -- saving this one's id made every resume
            # regenerate it and append its rows a second time.
            save_state_latent(args.dump_dir, {'concept_id': idx + 1}, 'latent')
    finally:
        # on failure, drop queued concepts; ones already running finish and are discarded
        executor.shutdown(wait=True, cancel_futures=True)


def generate_training(args, generate_args):
    dump_dir = args.dump_dir
    dump_dir = Path(dump_dir) / "generate"
    dump_dir.mkdir(parents=True, exist_ok=True)

    concept_path = args.concept_path
    num_base_instructions = args.num_base_instructions
    max_concepts = args.max_concepts

    set_seed(args.seed)
    all_concepts, all_refs = load_concepts(concept_path)

    # The resume watermark in STATE_FILE is a position in the selected list. A dump
    # written before SELECTED_CONCEPTS_FILE existed indexed a different, mixed-genre
    # list, so resuming it here would silently skip or repeat concepts.
    state = load_state(dump_dir)
    if state and not os.path.exists(os.path.join(dump_dir, SELECTED_CONCEPTS_FILE)):
        raise RuntimeError(
            f"{dump_dir} has a {STATE_FILE} but no {SELECTED_CONCEPTS_FILE}: it was written "
            f"by the old slice-then-filter concept selection and can't be resumed. Use a "
            f"fresh --dump_dir.")
    start_concept_id = state.get("concept_id", 0) if state else 0
    logger.warning(f"Starting concept index: {start_concept_id}")

    # Create a new OpenAI client (worker threads each make their own, see below).
    client = make_openai_client()
    num_workers = int(args.num_workers or 1)
    request_limiter = RequestLimiter(int(args.max_concurrent_requests)) \
        if args.max_concurrent_requests else None
    logger.warning(
        f"Generating with {num_workers} worker(s), max concurrent requests: "
        f"{args.max_concurrent_requests or f'uncapped (up to {64 * num_workers})'}.")

    # Load lm and tokenizer.
    model_name = model_name_map[all_refs[0].split("/")[3]]
    # Disabled: unused for --mode training now that create_train_df's instruction-mode
    # path (and the now-disabled global negative pool in DatasetFactory.__init__) no
    # longer call the local base LM at all -- only the gpt-4o-mini API client generates
    # anything. Loading this onto GPU was dead weight.
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name, torch_dtype=torch.bfloat16)
    model = None
    is_chat_model = True if model_name in CHAT_MODELS else False
    include_system_prompt = True if model_name == "meta-llama/Llama-3.1-8B-Instruct" else False
    # model = model.cuda()

    tokenizer =  AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    tokenizer.padding_side = "right"

    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        need_resize = True
    else:
        need_resize = False
    if model is not None and need_resize:
        model.resize_token_embeddings(len(tokenizer))

    # Init the dataset factory.
    dataset_factory = DatasetFactory(
        model, client, tokenizer, args.dataset_category, args.num_of_examples, args.output_length,
        dump_dir, use_cache=args.lm_use_cache, master_data_dir=args.master_data_dir,
        seed=args.seed, lm_model=args.lm_model, start_concept_id=start_concept_id, is_chat_model=is_chat_model,
        include_system_prompt=include_system_prompt,
        request_limiter=request_limiter, rate_limit_retries=RATE_LIMIT_RETRIES,
    )
    atexit.register(dataset_factory.save_cache)
    atexit.register(dataset_factory.reset_stats)

    # The one base-instruction pool every concept and every later stage builds on.
    base_instructions = load_or_create_base_instructions(
        dump_dir, dataset_factory.seed_instructions, num_base_instructions)
    # Held-out pool too, always: it costs no API calls, and writing it here means
    # flipping the inference config's steering_instructions_dist to "test" later never
    # requires regenerating, and the two pools cannot drift apart.
    load_or_create_base_instructions(
        dump_dir, dataset_factory.seed_instructions, num_base_instructions, dist="test")

    selected = select_text_concepts(
        all_concepts, all_refs, max_concepts, args.seed,
        getattr(args, "seed_concepts_dir", None), dataset_factory, str(dump_dir),
        concept16k_dir=getattr(args, "concept16k_dir", None))
    concepts = [(e["concept"], e["ref"]) for e in selected]
    concept_genres_map = {e["concept"]: e["genres"] for e in selected}
    if start_concept_id >= len(concepts):
        logger.warning(f"Datasets for all concepts have been generated. Exiting.")
        return

    remaining_ids = range(start_concept_id, len(concepts))
    only_one_concept = True if len(concepts) == 1 else False

    # Contrast pools for every remaining concept up front: one batched pass instead of
    # a round trip per concept, and the store is written here once rather than
    # concurrently by the workers below.
    text_ids = list(remaining_ids)
    contrastive_concepts_map = load_or_create_contrastive_concepts_batch(
        dataset_factory.lm_model, [concepts[i][0] for i in text_ids], dump_dir)

    thread_state = threading.local()

    def generate_concept(concept_id):
        # an httpx client is tied to the event loop it runs on, so each thread needs
        # its own; stats, cache and request limiter stay shared.
        if not hasattr(thread_state, "lm_model"):
            thread_state.lm_model = dataset_factory.lm_model.fork(make_openai_client())
        concept = concepts[concept_id][0]
        return dataset_factory.create_train_df(
            concept, concept_genres_map, base_instructions, contrastive_concepts_map[concept],
            output_length=args.output_length,
            only_one_concept=only_one_concept,
            lm_model=thread_state.lm_model,
            # seeded by concept rather than drawn from the global RNG, so contrast
            # assignment doesn't depend on thread timing or where a resume started
            rng=random.Random(f"{args.seed}:{concept_id}"),
        )

    # Counts concepts that actually produced data. The selected list is text-only, so
    # this now equals the loop index, but deriving it from the metadata written so far
    # keeps concept_id gap-free even if a save and the state watermark ever disagree --
    # it indexes the weight tensors and max_act lookups downstream.
    data_concept_id = count_metadata_entries(dump_dir)
    executor = ThreadPoolExecutor(max_workers=num_workers)
    futures = {i: executor.submit(generate_concept, i) for i in text_ids}
    try:
        # Workers finish in any order; results are saved strictly in concept order,
        # which the appended metadata/parquet, data_concept_id and the single
        # resume watermark in STATE_FILE all rely on.
        for concept_id in tqdm(remaining_ids, desc="Processing concept"):
            concept, ref = concepts[concept_id]
            genre = concept_genres_map[concept][0]
            # every stage shares one text-genre instruction pool, so anything else would
            # silently be paired with instructions from the wrong genre.
            assert genre == "text", f"non-text concept '{concept}' reached generation"

            current_df = futures.pop(concept_id).result()
            current_df["concept_id"] = data_concept_id

            # Save the generated DataFrame, metadata, and current state
            save(
                dump_dir, {"concept_id": concept_id + 1}, data_concept_id,
                concept, {concept: concept_genres_map[concept]},
                ref, "train", current_df, dataset_factory)
            data_concept_id += 1
            if data_concept_id == PREVIEW_CONCEPTS:
                save_preview(dump_dir, "train")
    finally:
        # on failure, drop queued concepts; ones already running finish and are discarded
        executor.shutdown(wait=True, cancel_futures=True)

    # fewer than PREVIEW_CONCEPTS text concepts in total: preview whatever there is
    if data_concept_id > 0 and not os.path.exists(os.path.join(dump_dir, PREVIEW_FILE)):
        save_preview(dump_dir, "train")

    logger.warning(f"Finished creating dataset.")


def save_dpo(
    dump_dir, concept_id, partition,
    current_df):
    # This function saves DataFrames per rank per partition (latent or steering)
    dump_dir.mkdir(parents=True, exist_ok=True)
    
    # Save DataFrame using Parquet
    rotation_freq = 500
    file_index = concept_id // rotation_freq if concept_id != -1 else 0
    if file_index == 0:
        df_path = os.path.join(dump_dir, f"{partition}_train_data.parquet")
    else:
        df_path = os.path.join(dump_dir, f"{partition}_train_data_{file_index}.parquet")
    if os.path.exists(df_path):
        existing_df = pd.read_parquet(df_path)
        combined_df = pd.concat([existing_df, current_df], ignore_index=True)
    else:
        combined_df = current_df
    combined_df.to_parquet(df_path, index=False)


def save_state_dpo(dump_dir, state, partition):
    dump_dir.mkdir(parents=True, exist_ok=True)
    # Save state
    state_path = os.path.join(dump_dir, f"{partition}_{STATE_FILE}")
    with open(state_path, "wb") as f:
        pickle.dump(state, f)


def load_state_dpo(dump_dir, partition):
    """
    Load the state from a file if it exists.
    """
    state_path = os.path.join(f"{dump_dir}/generate", f"{partition}_{STATE_FILE}")
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            return pickle.load(f)
    return None


def generate_dpo_training(args, inference_args):
    dump_dir = args.dump_dir
    dump_dir = Path(dump_dir) / "generate"
    args.data_dir = f"{args.dump_dir}/generate"
    # check the generate directory exists.
    if not os.path.exists(dump_dir):
        raise ValueError(f"Generate directory does not exist: {dump_dir}")
    # check the train_data.parquet exists.
    if not os.path.exists(os.path.join(dump_dir, "train_data.parquet")):
        raise ValueError(f"Train data does not exist: {os.path.join(dump_dir, 'train_data.parquet')}")

    concept_path = args.concept_path
    num_of_examples = args.num_of_examples
    max_concepts = args.max_concepts

    # Load and optionally shuffle concepts
    set_seed(int(args.seed))

    # Configure the logger per rank
    logger.setLevel(logging.WARNING)  # Set the logging level as desired

    # Create a logging formatter that includes the rank
    formatter = logging.Formatter(
        fmt=f'%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
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
    data_dir = args.data_dir
    num_of_examples = args.latent_num_of_examples
    metadata = load_metadata_flatten(data_dir)
    # Get list of all concept_ids
    concept_ids = list(range(len(metadata)))
    concepts = [metadata[i]["concept"] for i in concept_ids]
    
    # Load the state if it exists.
    state = load_state_dpo(args.dump_dir, "dpo")
    start_concept_id = state.get("concept_id", 0) if state else 0
    logger.warning(f"Starting concept index: {start_concept_id}")
    if start_concept_id >= len(concept_ids):
        logger.warning(f"Datasets for all concepts have been generated. Exiting.")
        return

    # Create a new OpenAI client.
    client = make_openai_client()

    # Init the dataset factory.
    dataset_factory = DatasetFactory(
        None, client, None, args.dataset_category, num_of_examples, int(args.output_length), 
        dump_dir, use_cache=args.lm_use_cache, master_data_dir=args.master_data_dir,
        seed=int(args.seed), lm_model=args.lm_model, start_concept_id=start_concept_id, 
        is_inference=True, is_dpo=True, concepts=concepts,
        disable_local_model=args.disable_local_model,
    )
    atexit.register(dataset_factory.save_cache)
    atexit.register(dataset_factory.reset_stats)
    
    # get negative and do nothing on them just renaming.
    existing_df = pd.read_parquet(os.path.join(dump_dir, "train_data.parquet"))
    existing_df = existing_df.rename(
        columns={
            'output': 'winning_output', 
        }
    )

    model, tokenizer = None, None
    if args.keep_orig_axbench_format:
        # we need to load the model and the tokenizer back in this case to generate
        # the same data distribution as AxBench original to keep the comparison fair.
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16)
        is_chat_model = True if args.model_name in CHAT_MODELS else False
        include_system_prompt = True if args.model_name in HAS_SYSTEM_PROMPT_MODELS else False
        model = model.cuda()
        tokenizer =  AutoTokenizer.from_pretrained(args.model_name, model_max_length=512)
        tokenizer.padding_side = "right"

    progress_bar = tqdm(range(start_concept_id, len(metadata)), desc="Processing concept")
    for start_idx in progress_bar:
        concept_id = metadata[start_idx]["concept_id"]
        concept = metadata[start_idx]["concept"]
        print(f"Generating for concept: {concept}...")
        current_df = existing_df[existing_df["concept_id"] == concept_id].copy()
        dpo_df = dataset_factory.create_dpo_df(
            current_df,
            output_length=int(args.output_length),
            batch_size=int(args.inference_batch_size),
            model=model,
            tokenizer=tokenizer,
            keep_orig_axbench_format=args.keep_orig_axbench_format,
            steer_data_type = args.steer_data_type            
        )

        save_dpo(dump_dir, concept_id, 'dpo', dpo_df)
        logger.warning(f"Saved dpo dataset for concept {concept_id} to dpo_train_data.parquet")
        # After processing, save state
        current_state = {'concept_id': concept_id}
        save_state_dpo(dump_dir, current_state, 'dpo')

    logger.warning(f"Finished creating DPO dataset.")


def main():
    custom_args = [
        {
            'args': ['--mode'],
            'kwargs': {
                'type': str,
                'default': "training",
                'help': 'The generation mode.'
            }
        }
    ]

    generate_args = DatasetArgs(custom_args=custom_args, section="generate")
    inference_args = DatasetArgs(custom_args=custom_args, section="inference")
    logger.warning("Generating datasets with the following configuration:")
    logger.warning(generate_args)

    if generate_args.mode == "training":
        generate_training(generate_args, inference_args)
    elif generate_args.mode == "latent":
        generate_latent(generate_args, inference_args)   
    elif generate_args.mode == "dpo_training":
        generate_dpo_training(generate_args, inference_args)
    else:
        raise ValueError(f"Invalid mode: {generate_args.mode}")


if __name__ == "__main__":
    main()

