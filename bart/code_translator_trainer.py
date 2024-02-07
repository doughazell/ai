# 29/1/24 DH: Created from 'summarize.py'

import logging
import os
import sys
import datasets
from datasets import load_dataset
import evaluate
import nltk
import numpy as np

import transformers
from transformers import (
  HfArgumentParser,
  Seq2SeqTrainingArguments,
  set_seed,
  AutoConfig,
  AutoTokenizer,
  AutoModelForSeq2SeqLM,
  DataCollatorForSeq2Seq,
  Seq2SeqTrainer,
)

# 30/1/24 DH:
from code_translator_data import Seq2SeqModelData
from code_translator_trainer_arguments import DataTrainingArguments, ModelArguments, summarization_name_mapping

# 30/1/24 DH: Reverse the dict: 'log_levels' (See 'Seq2SeqTrainingArguments.to_dict()' for "k,v" checking)
log_levels_rev = {v: k for k, v in transformers.utils.logging.log_levels.items()}

logger = logging.getLogger(__name__)

def trainSeq2SeqLM(modelName) -> Seq2SeqModelData:
    
    ################################################################################
    # 1/6) Config
    ################################################################################

    print()
    print("1/6) Config")
    print()

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    # 30/1/24 DH:
    sys.argv.append("summarize.json")

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 30/1/24 DH:
    model_args.model_name_or_path = modelName

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    print()
    print("  1/6) REMOVED: 'logger.info(f\"Training/evaluation parameters {training_args}\")' ")
    print()

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
            
    # Set seed before initializing model.
    set_seed(training_args.seed)

    ################################################################################
    # 2/6) Dataset preprocessing
    ################################################################################

    print()
    print("2/6) Dataset preprocessing")
    print()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.
    
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    #
    # GET: 'column_names', 'text_column', 'summary_column'
    #
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        column_names = raw_datasets["train"].column_names

    # Get the column names for input/target - DECLARED AS A DICT ABOVE
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)

    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    ################################################################################
    # 3/6) Loading Model + Tokenizer
    ################################################################################

    print()
    print("3/6) Loading Model + Tokenizer")
    print()

    # 30/1/24 DH: Prevent 'info' logging for this section
    transformers.utils.logging.set_verbosity_error()
    logRet = transformers.utils.logging.get_verbosity()
    logRetName = log_levels_rev[logRet]

    print()
    print(f"  3/6) ALTERED LOGGING TO: '{logRetName}'")
    print()
    
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    print()
    print("  3/6) AutoConfig.from_pretrained()")
    print()
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    print()
    print("  3/6) AutoTokenizer.from_pretrained()")
    print()
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    print()
    print("  3/6) AutoModelForSeq2SeqLM.from_pretrained()")
    print()
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # 30/1/24 DH: Reset logging back to 'info'
    transformers.utils.logging.set_verbosity_info()
    logRet = transformers.utils.logging.get_verbosity()
    logRetName = log_levels_rev[logRet]
    
    print()
    print(f"  3/6) ALTERED LOGGING TO: '{logRetName}'")
    print()
    
    print()
    print("  3/6) Resizing embeddings")
    print()

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
            
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    ################################################################################
    # 4/6) Training token mapping
    ################################################################################

    print()
    print("4/6) Training token mapping")
    print()

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for "
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        # remove pairs where at least one record is None

        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])

        inputs = [prefix + inp for inp in inputs]

        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        with training_args.main_process_first(desc="train dataset map pre-processing"):
            print()
            print("  4/6) Running tokenizer on train dataset")
            print()
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    # Metrics
    metric = evaluate.load("rouge", cache_dir=model_args.cache_dir)

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        #Replace -100s used for padding as we can't decode them"
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    ################################################################################
    # 5/6) Training
    ################################################################################

    print()
    print("5/6) Training")
    print()

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    print()
    print(f"  5/6) {trainer.__class__}::compute_metrics() = {trainer.compute_metrics}")
    print()

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        
        # 30/1/24 DH: === CALL: 'Seq2SeqTrainer.train()'  ===
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
                    
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    
    trainer.create_model_card(**kwargs)

    ################################################################################
    # 6/6) Return objects via API
    ################################################################################

    print()
    print("6/6) Return objects via API")
    print()

    # 30/1/24 DH:
    seq2seqModelData = Seq2SeqModelData()

    seq2seqModelData.model_name_or_path = model_args.model_name_or_path
    seq2seqModelData.dataset_name       = data_args.dataset_name
    seq2seqModelData.dataset_config     = data_args.dataset_config_name

    seq2seqModelData.bart_config                     = config
    seq2seqModelData.bart_tokenizer_fast             = tokenizer
    seq2seqModelData.bart_for_conditional_generation = model
    seq2seqModelData.data_collator_for_seq2seq       = data_collator
    seq2seqModelData.seq2seq_trainer                 = trainer

    seq2seqModelData.model_arguments            = model_args
    seq2seqModelData.data_training_arguments    = training_args
    seq2seqModelData.seq2seq_training_arguments = data_args

    # 31/1/24 DH: Reset log level outside the Trainer API
    transformers.utils.logging.set_verbosity( transformers.utils.logging._default_log_level )
    logRet = transformers.utils.logging.get_verbosity()
    logRetName = log_levels_rev[logRet]
    
    print()
    print(f"  6/6) RESET LOGGING TO: '{logRetName}'")
    print()
    
    return seq2seqModelData

