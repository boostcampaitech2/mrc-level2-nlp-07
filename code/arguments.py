from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
<<<<<<< HEAD
        default="klue/roberta-large",
=======
        default="./models/klue15_hard2/checkpoint-14500/",
>>>>>>> 1ec842f4bafe0c045d611551db1f2b9ac8e0f169
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        }
    )
    tokenizer_name: Optional[str] = field(
        default="klue/roberta-large",
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
<<<<<<< HEAD
        default="/opt/ml/mrc-level2-nlp-07/data/train_dataset",
=======
        default="../data/test_dataset/",
>>>>>>> 1ec842f4bafe0c045d611551db1f2b9ac8e0f169
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    num_clusters: int = field(
        default=32, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
<<<<<<< HEAD
        default=10,
=======
        default=15,
>>>>>>> 1ec842f4bafe0c045d611551db1f2b9ac8e0f169
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=False, metadata={"help": "Whether to build with faiss"}
    )

@dataclass
class TrainingArguments:
    """
    Arguments pertaining to which hyperparameters would trainer takes
    """
    output_dir: str = field(
        default='./models/xlm/',
        metadata={
            "help": "Path to save training model"
        },
    )
    num_train_epochs: int = field(
        default=5,
        metadata={
            "help": "Number of epochs for training"
        },
    )

    save_total_limit: int = field(
        default=2,
        metadata={
            "help": "Number of checkpoint saved while training"
        },
    )
    save_steps: int = field(
        default=500,
        metadata={
            "help": "Save and delete steps while traininig"
        },
    )
    eval_steps: int = field(
        default=500,
        metadata={
            "help":"Steps for evaluation while training, same as save_steps"
        },
    )
    learning_rate: float = field(
        default=1e-6,
        metadata={
            "help": "The initial learning rate"
        },
    )
    evaluation_strategy: str = field(
        default='epoch',
        metadata={
            "help": "The evaluation strategy to adopt during training"
        },
    )

    use_elastic: bool = field(
        default=False, metadata={"help": "Whether to use elastic search"}
    )

