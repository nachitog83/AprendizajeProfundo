import argparse
import gzip
import json
import logging
import mlflow
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from .dataset import MeliChallengeDataset
from .utils import PadSequences


logging.basicConfig(
    format="%(asctime)s: %(levelname)s - %(message)s",
    level=logging.INFO
)

class CNNClassifier(nn.Module):
    def __init__(self,
                 pretrained_embeddings_path,
                 token_to_index,
                 n_labels,
                 vector_size=300,
                 filters_count=100,      ###### FEATURES LERNERS
                 filters_width=[2,3,4], ###### WORDS PER CONVOLUTION
                 dimensions=128,
                 freeze_embedings=True):

        super().__init__()
        with gzip.open(token_to_index, "rt") as fh:
            token_to_index = json.load(fh)
        embeddings_matrix = torch.randn(len(token_to_index), vector_size)
        embeddings_matrix[0] = torch.zeros(vector_size)
        with gzip.open(pretrained_embeddings_path, "rt") as fh:
            next(fh)
            for line in fh:
                word, vector = line.strip().split(None, 1)
                if word in token_to_index:
                    embeddings_matrix[token_to_index[word]] =\
                        torch.FloatTensor([float(n) for n in vector.split()])

        self.embeddings = nn.Embedding.from_pretrained(embeddings_matrix,
                                                       freeze=freeze_embedings,
                                                       padding_idx=0)
        self.convs = []
        for filter in filters_width:
            self.convs.append(
                nn.Conv1d(vector_size, filters_count, filter)
            )

        self.convs = nn.ModuleList(self.convs)
        self.fc = nn.Linear(filters_count * len(filters_width), dimensions)
        self.output = nn.Linear(dimensions, n_labels)
        self.vector_size = vector_size
        self.filters_count = filters_count
        self.filters_width = filters_width
        self.dimensions=dimensions
   
    @staticmethod
    def conv_global_max_pool(x, conv):
        return F.relu(conv(x).transpose(1, 2).max(1)[0])
    
    def forward(self, x):
        x = self.embeddings(x).transpose(1, 2)  # Conv1d takes (batch, channel, seq_len)
        x = [self.conv_global_max_pool(x, conv) for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = F.relu(self.fc(x))
        x = self.output(x) # Softmax is applied in Cross Entropy Loss
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data",
                        help="Path to the the training dataset",
                        required=True)
    parser.add_argument("--token-to-index",
                        help="Path to the the json file that maps tokens to indices",
                        required=True)
    parser.add_argument("--pretrained-embeddings",
                        help="Path to the pretrained embeddings file.",
                        required=True)
    parser.add_argument("--language",
                        help="Language working with",
                        required=True)
    parser.add_argument("--test-data",
                        help="If given, use the test data to perform evaluation.")
    parser.add_argument("--validation-data",
                        help="If given, use the validation data to perform evaluation.")
    parser.add_argument("--embeddings-size",
                        default=300,
                        help="Size of the vectors.",
                        type=int)
    parser.add_argument("--epochs",
                        help="Number of epochs",
                        default=3,
                        type=int)
    parser.add_argument("--batch-size",
                        help="Batch size",
                        default=128,
                        type=int)
    parser.add_argument("--random-buffer-size",
                        help="Random buffer size",
                        default=2048,
                        type=int)
    parser.add_argument("--freeze-embedings",
                        help="Freeze embeddings?",
                        default=True)
    parser.add_argument("--lr",
                        help="Learning Rate regularization",
                        default=1e-3,
                        type=float)
    parser.add_argument("--weight-decay",
                        help="Weight Decay",
                        default=1e-5,
                        type=float)
    parser.add_argument("--filters-count",
                        help="Number of filters (feature lerners)",
                        default=100,
                        type=int)
    parser.add_argument("--filters-width",
                        help="Filters width (words per convolution)",
                        nargs="+",
                        default=[2, 3, 4],
                        type=int)
    parser.add_argument("--dimensions",
                        help="Number of Layer's dimensions",
                        default=128,
                        type=int)

    args = parser.parse_args()

    assert args.batch_size > 0
    assert args.random_buffer_size > 0

    pad_sequences = PadSequences(
        pad_value=0,
        max_length=max(args.filters_width),
        min_length=1
    )

    logging.info("Building training dataset")
    train_dataset = MeliChallengeDataset(
        dataset_path=args.train_data,
        random_buffer_size=2048  # This can be a hypterparameter
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,  # This can be a hyperparameter
        shuffle=False,
        collate_fn=pad_sequences,
        drop_last=False
    )

    if args.validation_data:
        logging.info("Building validation dataset")
        validation_dataset = MeliChallengeDataset(
            dataset_path=args.validation_data,
            random_buffer_size=1
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=pad_sequences,
            drop_last=False
        )
    else:
        validation_dataset = None
        validation_loader = None

    if args.test_data:
        logging.info("Building test dataset")
        test_dataset = MeliChallengeDataset(
            dataset_path=args.test_data,
            random_buffer_size=1
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=pad_sequences,
            drop_last=False
        )
    else:
        test_dataset = None
        test_loader = None

    mlflow.set_experiment(f"diplodatos.{args.language}.CNN")

    with mlflow.start_run():
        logging.info("Starting experiment")
        # Log all relevent hyperparameters
        mlflow.log_params({
            "model_type": "Convolutional Neural Network",
            "embeddings": args.pretrained_embeddings,
            "embeddings_size": args.embeddings_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "random_buffer_size": args.random_buffer_size,
            "freeze_embedings": args.freeze_embedings,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "filters_count": args.filters_count,
            "filters_width": args.filters_width,
            "dimensions": args.dimensions
        })
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logging.info("Building classifier")
        model = CNNClassifier(
            pretrained_embeddings_path=args.pretrained_embeddings,
            token_to_index=args.token_to_index,
            n_labels=train_dataset.n_labels,
            vector_size=args.embeddings_size,
            freeze_embedings=args.freeze_embedings,
            filters_count=args.filters_count,
            filters_width=args.filters_width,
            dimensions=args.dimensions
        )
        model = model.to(device)
        loss = nn.CrossEntropyLoss() ### Applies softmax()
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        logging.info(model)
        logging.info(loss)
        logging.info(optimizer)

        logging.info("Training classifier")
        for epoch in trange(args.epochs):
            model.train()
            running_loss = []
            for idx, batch in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                data = batch["data"].to(device)
                target = batch["target"].to(device)
                output = model(data)
                loss_value = loss(output, target)
                loss_value.backward()
                optimizer.step()
                running_loss.append(loss_value.item())
            mlflow.log_metric("train_loss", sum(running_loss) / len(running_loss), epoch)

            if validation_dataset:
                logging.info("Evaluating model on validation")
                model.eval()
                running_loss = []
                targets = []
                predictions = []
                with torch.no_grad():
                    for batch in tqdm(validation_loader):
                        data = batch["data"].to(device)
                        target = batch["target"].to(device)
                        output = model(data)
                        running_loss.append(
                            loss(output, target).item()
                        )
                        targets.extend(batch["target"].numpy())
                        predictions.extend(output.argmax(axis=1).detach().cpu().numpy())
                    mlflow.log_metric("validation_loss", sum(running_loss) / len(running_loss), epoch)
                    mlflow.log_metric("validation_bacc", balanced_accuracy_score(targets, predictions), epoch)

        if test_dataset:
            logging.info("Evaluating model on test")
            model.eval()
            running_loss = []
            targets = []
            predictions = []
            with torch.no_grad():
                for batch in tqdm(test_loader):
                    data = batch["data"].to(device)
                    target = batch["target"].to(device)
                    output = model(data)
                    running_loss.append(
                        loss(output, target).item()
                    )
                    targets.extend(batch["target"].numpy())
                    predictions.extend(output.argmax(axis=1).detach().cpu().numpy())
                mlflow.log_metric("test_loss", sum(running_loss) / len(running_loss), epoch)
                mlflow.log_metric("test_bacc", balanced_accuracy_score(targets, predictions), epoch)
