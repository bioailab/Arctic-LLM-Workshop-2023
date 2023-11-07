import transformers as tr
from datasets import load_dataset
import os

# PORT = "8819"
# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = PORT  # any free port
# os.environ["RANK"] = "0"
# os.environ["LOCAL_RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"  # Only one system is used for this example


rotten = load_dataset("rotten_tomatoes")  # load the dataset
# model_checkpoint = "t5-small"
model_checkpoint = "t5-base"

# model = tr.AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)  # Alternative way to initialize the model
model = tr.T5ForConditionalGeneration.from_pretrained(model_checkpoint)  # Initialize the model

# tokenizer = tr.AutoTokenizer.from_pretrained(model_checkpoint)  # Alternative way to initialize the tokenizer
tokenizer = tr.T5Tokenizer.from_pretrained(model_checkpoint)  # Initialize the tokenizer

classes = {0: "negative", 1: "positive"}  # Sentiment classes

def map_fn(data):  # Function to tokenize the dataset
    # Convert the dataset to a tokenized dataset, and remove the columns that are not needed anymore
    return tokenizer(
            data['text'],  # Tokenize the text
            text_target=[classes[label] for label in data['label']],  # Convert 0/1 to "negative"/"positive" text labels
            truncation=True,  # Truncate the inputs if they are too long
            padding=True,  # Pad the inputs if they are too short
            return_tensors='np'  # Return NumPy tensors
        )

# Tokenize the dataset
rotten_tokenized = rotten.map(  # Maps the tokenize function to each split of the dataset
    map_fn, 
    batched=True,  # Batch the outputs
    remove_columns=['text', 'label']  # Remove the untokenized columns
)  # Tokenize the dataset

# Define params
checkpoint_dir = "checkpoints"  # Directory to save the checkpoints to
run_name = "t5-small-rotten-tomatoes-deepspeed"  # A name for the current training run
epochs = 3  # Number of training epochs
batch_size = 128  # Batch size
optimizer = "adamw_torch"  # Optimizer to use. Adam with weight decay is a good standard choice
# tensorboard_dir = "tensorboard"  # Directory to save TensorBoard logs to

# Create the directories
checkpoint_path = os.path.join(checkpoint_dir, run_name)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Initialize trainer
data_collator = tr.DataCollatorWithPadding(tokenizer=tokenizer)  # Initialize the data collator (data collator does the same thing as a data loader in PyTorch)

# Define the ZeRO config
# Define the DeepSpeed zero optimization config
zero_config = {
    "train_batch_size": "auto",  # train_batch_size is the global batch size
    "train_micro_batch_size_per_gpu": "auto",  # train_micro_batch_size_per_gpu is the per-GPU batch size
    # train_batch_size = train_micro_batch_size_per_gpu * gradient_accumulation_steps * number of GPUs

    # ZeRO parameters
    "zero_optimization": {
        # "stage": 2,  # Enable ZeRO Stage 2
        # "offload_optimizer": {"device": "cpu", "pin_memory": True},  # Offload the optimizer to the CPU  
        
        'stage': 3,
        'stage3_gather_16bit_weights_on_model_save': True,  # Gather 16-bit weights to the full precision model during save (allows saving mixed precision models)

        "contiguous_gradients": True,  # Enable contiguous gradients, which improves performance by reducing memory fragmentation
        "overlap_comm": True,
    },

    # Optimizer
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto",
            "torch_adam": True,
        },
    },

    # Scheduler
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
        }
    }
}

# Define the training arguments
training_args = tr.TrainingArguments(
    checkpoint_path,
    num_train_epochs=epochs,
    
    per_device_train_batch_size=batch_size,
    # per_device_train_batch_size=batch_size * 2,  # Double the batch size

    deepspeed=zero_config,  # Pass in the ZeRO config
    # fp16=True,  # Use FP16 training
    # bf16=True,  # Use BF16 training (only on Ampere GPUs like the A100)
)

# Initialize the trainer
data_collator = tr.DataCollatorWithPadding(tokenizer=tokenizer)  # Initialize the data collator (data collator does the same thing as a data loader in PyTorch)
trainer = tr.Trainer(
    model,
    training_args,
    train_dataset=rotten_tokenized['train'],  # Pass in the tokenized training dataset
    eval_dataset=rotten_tokenized['validation'],  # Pass in the tokenized validation dataset
    data_collator=data_collator,  # Pass in the data collator
    tokenizer=tokenizer,  # Pass in the tokenizer
)

trainer.train()
trainer.save_model()

# Pause and wait for the user to ctrl + c
input("Press ctrl + c to exit")