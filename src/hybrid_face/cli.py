"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
         hybrid_face = hybrid.cli:run

Then run `python setup.py install` which will install the command `hybrid_face`
inside your current environment.
Besides console scripts, the header (i.e. until logger...) of this file can
also be used as template for Python modules.

"""
import argparse
import sys
from pathlib import Path

from PIL import Image

from hybrid_face import __version__
from hybrid_face.hybrid_merge import hybrid_merge

sigma_dict = {"far": 0.005, "balanced": 0.002, "near": 0.0005}


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Command-line tool for creating hybrid images")
    parser.add_argument(
        "--version",
        action="version",
        version="hybrid_face {ver}".format(ver=__version__),
    )

    parser.add_argument(
        "-n",
        "--near",
        action="store",
        type=Path,
        required=True,
        help="Path to the image that should be seen from anear",
        dest="near_image_path",
        metavar="NEAR_IMAGE",
    )

    parser.add_argument(
        "-f",
        "--far",
        action="store",
        type=Path,
        required=True,
        help="Path to the image that should be seen from afar",
        dest="far_image_path",
        metavar="FAR_IMAGE",
    )

    parser.add_argument(
        "--emphasis",
        action="store",
        choices=["near", "far", "balanced"],
        dest="emphasis",
        default="balanced",
        help="Choose whether the near or far image should be emphasized",
    )

    parser.add_argument(
        "-o",
        "--output",
        action="store",
        type=Path,
        dest="output_file",
        default=None,
        help='Output file for the resulting image (e.g. "result.png")',
        metavar="OUTPUT",
    )

    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        dest="show",
        help="Set this flag if you want to display the image (and not necessarily save it)",
    )

    return parser.parse_args(args)


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)

    sigma = sigma_dict[args.emphasis]
    near_image = Image.open(args.near_image_path)
    far_image = Image.open(args.far_image_path)
    hybrid_image = hybrid_merge(near_image, far_image, sigma)

    if args.show:
        hybrid_image.show()

    if args.output_file is not None:
        hybrid_image.save(args.output_file)


def run():
    """Entry point for console_scripts"""
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
