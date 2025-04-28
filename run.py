#!/usr/bin/env python3
import argparse
import os

# import torch

__version__ = "1.0.0"
DEFAULT_MODEL_WEIGHTS_OUTPUT_DIR = "output"
DEFAULT_MODEL_WEIGHTS_FNAME = "model_weights.pth"


def train(args):
    print(
        f"Training for {args.epochs} epochs with batch size {args.batch_size} and lr {args.learning_rate}"
    )
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, DEFAULT_MODEL_WEIGHTS_FNAME)
    # model = build_model()
    # loader = get_data_loader(args.train_data, args.batch_size)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # for epoch in range(args.epochs):
    #     for batch in loader:
    #         optimizer.zero_grad()
    #         outputs = model(batch["input"])
    #         loss = compute_loss(outputs, batch["target"])
    #         loss.backward()
    #         optimizer.step()
    # torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def infer(args):
    print(f"\nInference request received:\n{args.input}\n")
    print(f"Training logs start..")
    print(f"Loading model from {args.model_path}")
    # model = build_model()
    # model.load_state_dict(torch.load(args.model_path))
    # model.eval()
    # input_data = load_input(args.input)
    # output = model(input_data)
    print(f"Training logs end!")
    output = "TODO"
    print(f"\nInference output:\n{output}\n")


def info(args):
    print(f"Agent script for training and inference. Version: {__version__}")


def main():
    parser = argparse.ArgumentParser(
        prog="run.py", description="Agent script for training and inference"
    )
    subparsers = parser.add_subparsers(
        title="subcommands", dest="command", required=True
    )
    # Training args
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    train_parser.add_argument(
        "--train-data", type=str, default="training_data", help="Path to training data"
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_MODEL_WEIGHTS_OUTPUT_DIR,
        help="Directory to save model",
    )
    # Inference args
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_WEIGHTS_FNAME,
        help="Path to saved model",
    )
    infer_parser.add_argument(
        "--input", type=str, required=True, help="Input for inference"
    )
    # Info subcommand
    subparsers.add_parser("info", help="Show script version")
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "infer":
        infer(args)
    elif args.command == "info":
        info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
