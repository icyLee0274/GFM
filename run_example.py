from argparse import ArgumentParser
from examples import *
import torch, os

from examples.hypercube import Hypercube


def main():
    parser = ArgumentParser()

    parser.add_argument("--gen_sample", action="store_true")
    parser.add_argument("--n_sample", type=int, default=10000)
    parser.add_argument("--n_epoch", type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--n_gen", type=int, default=1000)
    parser.add_argument("--n_step", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", type=str, default='vanilla')
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--output", type=str, default="example-out")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--gen", action="store_true")
    parser.add_argument("--test", action="store_true")

    parser.add_argument("example", type=str, help="Example to run")

    args = parser.parse_args()
    args.output = os.path.join(args.output, args.example)

    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(args.device)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    match args.example:
        case "hypercube2":
            ex = Hypercube(args, 2)
        case "hypercube3":
            ex = Hypercube(args, 3)
        case "hypercube10":
            ex = Hypercube(args, 10)
        case _:
            raise RuntimeError(f"Unknown example: {args.example}.")

    if args.train:
        ex.train()
        torch.save(ex.velocity, os.path.join(args.output, f"velocity.pt"))
        if ex.gen_sample:
            torch.save(ex.true_samples, os.path.join(args.output, f"true_samples.pt"))
    elif args.gen:
        ex.velocity = torch.load(os.path.join(args.output, f"velocity.pt"))
        if ex.verbose: print("Loaded model.")
        ex.generate()
    elif args.test:
        ex.init_training()
        ex.velocity = torch.load(os.path.join(args.output, f"velocity.pt"))
        if ex.verbose: print("Loaded model.")
        ex.generate_test()


if __name__ == "__main__":
    main()
