import sys
from argparse import REMAINDER, ArgumentParser

from optimum.habana.distributed import DistributedRunner


def parse_args():
    """
    Helper function parsing the command line options.
    @retval ArgumentParser
    """
    parser = ArgumentParser(
        description=(
            "Habana Gaudi distributed inference launch helper utility that will spawn up multiple distributed"
            " processes."
        )
    )

    # Optional arguments for the launch helper
    parser.add_argument("--world_size", type=int, default=1, help="Number of HPUs to use (1, 4 or 8)")
    parser.add_argument("--hostfile", type=str, default=None, help="Path to the file where hosts are specified.")
    parser.add_argument("--use_mpi", action="store_true", help="Use MPI for distributed inference")
    parser.add_argument("--use_deepspeed", action="store_true", help="Use DeepSpeed for distributed inference")

    # positional
    parser.add_argument(
        "inference_script",
        type=str,
        help=(
            "The full path to the single HPU inference "
            "program/script to be launched in parallel, "
            "followed by all the arguments for the "
            "inference script."
        ),
    )

    # rest from the training program
    parser.add_argument("inference_script_args", nargs=REMAINDER)

    return parser.parse_args()


def main():
    args = parse_args()

    # Patch sys.argv
    sys.argv = [args.inference_script] + args.inference_script_args
    # Handle the case where arguments contain whitespaces
    argv = ['"{}"'.format(arg) if " " in arg and arg[0] != '"' and arg[-1] != '"' else arg for arg in sys.argv]
    command_list = [" ".join(argv)]

    distributed_runner = DistributedRunner(
        command_list=command_list,
        world_size=args.world_size,
        hostfile=args.hostfile,
        use_mpi=False,
        use_deepspeed=args.use_deepspeed,
    )

    ret_code = distributed_runner.run()
    sys.exit(ret_code)


if __name__ == "__main__":
    main()
