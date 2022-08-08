'''ddmd run'''
import os
import argparse
import ddmd
import ddmd.scripts.run_md
import ddmd.scripts.run_ml
import ddmd.scripts.run_infer
import ddmd.scripts.run_ddmd
# from ddmd.utils import parse_args

def main(): 
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version='ddmd '+ddmd.__version__)

    modules = [
        ddmd.scripts.run_md, 
        ddmd.scripts.run_ml, 
        ddmd.scripts.run_infer, 
        ddmd.scripts.run_ddmd,
    ]

    subparsers = parser.add_subparsers(title='Choose a command')
    subparsers.required = 'True'

    def get_str_name(module):
        return os.path.splitext(os.path.basename(module.__file__))[0]

    for module in modules:
        this_parser = subparsers.add_parser(get_str_name(module), description=module.__doc__)
        # module.add_args(this_parser)
        # print(this_parser)
        this_parser.set_defaults(func=module.main)
        this_parser.add_argument(
        "-c", "--config", help="YAML config file", type=str, required=True
        )

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()

