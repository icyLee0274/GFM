from argparse import ArgumentParser
from examples import *
import torch, os

from examples.hypercube import Hypercube
from examples.toy_2d import Toy2D


def main():
    parser = ArgumentParser()

    parser.add_argument("--gen_sample", action="store_true")
    parser.add_argument("--norm", type=str, choices=["L2", "Linf"], default="Linf")
    parser.add_argument("--n_sample", type=int, default=10000)
    parser.add_argument("--n_epoch", type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--n_gen", type=int, default=1000)
    parser.add_argument("--n_step", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1145)
    parser.add_argument("--method", type=str, default='vanilla')
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--output", type=str, default="example-out")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--gen", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--interior_point", type=str, default=None)

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
        case "hypercube6":
            ex = Hypercube(args, 6)
        case "hypercube10":
            ex = Hypercube(args, 10)
        case "polytope_2d":
            ex = Polytope2D(args)
        case "polytope2":
            ex = Polytope(args, 2)
        case "polytope6":
            ex = Polytope(args, 6)
        case "polytope10":
            ex = Polytope(args, 10)
        case "qc3":
            ex = QC(args, 3)
        case "qc6":
            ex = QC(args, 6)
        case "toy2d":
            ex = Toy2D(args)
        case _:
            raise RuntimeError(f"Unknown example: {args.example}.")

    if args.train:
        ex.train()
        ex.save_model()
        if ex.gen_sample:
            torch.save(ex.true_samples, os.path.join(args.output, f"true_samples.pt"))
    elif args.gen:
        ex.load_model()
        if ex.verbose: print("Loaded model.")
        ex.generate()
    elif args.test:
        ex.init_training()
        ex.load_model()
        if ex.verbose: print("Loaded model.")
        ex.generate_test()


if __name__ == "__main__":
    main()
