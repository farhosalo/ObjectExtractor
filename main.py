import argparse
import filetype
from enum import Enum
import os

from ObjectExtractor.ObjectExtractorGDino import ObjectExtractorGDino
from ObjectExtractor.ObjectExtractorOwlVit import ObjectExtractorOwlVit
from ObjectExtractor import AbstractObjectExtractor
import Configuration
import logging

logging.basicConfig(level=logging.INFO)


class SupportedFileTypes(Enum):
    UNKNOWN = 0
    VIDEO = 1
    IMAGE = 2


def GetFileType(filename: str):
    if filename is None or not os.path.isfile(filename):
        return SupportedFileTypes.UNKNOWN

    mime = filetype.guess_mime(filename)
    if mime is None:
        return SupportedFileTypes.UNKNOWN

    if mime is not None:
        if mime.startswith("image/"):
            return SupportedFileTypes.IMAGE
        elif mime.startswith("video/"):
            return SupportedFileTypes.VIDEO

    return SupportedFileTypes.UNKNOWN


def extract(fileName: str, extractor: AbstractObjectExtractor):
    fileType = GetFileType(fileName)

    if fileType == SupportedFileTypes.VIDEO:
        extractor.extractFromVideo(fileName)
    elif fileType == SupportedFileTypes.IMAGE:
        extractor.extractFromImage(fileName)
    else:
        logging.error("Input file format is not supported")


def main():
    parser = argparse.ArgumentParser(
        prog="ObjectExtractor",
        description="Extracts objects from an image, video or directory",
    )

    parser.add_argument(
        "-d",
        "--device",
        default="mps",
        choices=["cuda", "mps", "cpu"],
        help="Selects the accelerating device, default is cpu, cuda for NVIDIA GPU and mps for Apple GPU",
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the input image or video"
    )
    parser.add_argument(
        "-c",
        "--classes",
        required=True,
        nargs="+",
        type=str,
        help='Space separated list of classes to extract (e.g. "traffic sign" "bus")',
    )
    args = parser.parse_args()

    underlyingModel = Configuration.config.get("UNDERLYING_MODEL")
    if underlyingModel is None or underlyingModel not in Configuration.UNDERLYING_MODEL:
        raise ValueError(
            f"Underlying model must be one of {Configuration.UNDERLYING_MODEL}, got {underlyingModel}"
        )

    minimumHeight = Configuration.config.get("MINIMUM_HEIGHT")
    if (
        minimumHeight is None
        or not isinstance(minimumHeight, int)
        or minimumHeight <= 0
    ):
        minimumHeight = 24
        logging.warning(
            f"Invalid minimum height value in configuration, defaulting to {minimumHeight}"
        )

    minimumWidth = Configuration.config.get("MINIMUM_WIDTH")
    if minimumWidth is None or not isinstance(minimumWidth, int) or minimumWidth <= 0:
        minimumWidth = 24
        logging.warning(
            f"Invalid minimum width value in configuration, defaulting to {minimumWidth}"
        )

    outputPath = Configuration.config.get("OUTPUT_PATH")
    if outputPath in [None, ""] or not isinstance(outputPath, str):
        outputPath = "Output"
        logging.warning(
            f"Invalid output path in configuration, defaulting to {outputPath}"
        )

    extractor: AbstractObjectExtractor

    if underlyingModel == "GroundingDINO":
        extractor = ObjectExtractorGDino(
            device=args.device,
            outputPath=outputPath,
            classes=args.classes,
        )
    elif underlyingModel == "OwlVit":
        extractor = ObjectExtractorOwlVit(
            device=args.device,
            outputPath=outputPath,
            classes=args.classes,
        )
    else:
        raise ValueError(
            f"Underlying model must be one of {Configuration.UNDERLYING_MODEL}, got {appConfig['UNDERLYING_MODEL']}"
        )
    logging.info(f"Minimum object size: {minimumWidth} x {minimumHeight}")
    extractor.setMinimumObjectSize(minimumWidth, minimumHeight)

    input = os.fsencode(args.input)
    if not os.path.exists(input):
        logging.fatal("Input path does not exist")
        return

    if os.path.isfile(input):
        extract(input.decode("utf-8"), extractor)
    else:
        for file in os.listdir(input):
            f = os.fsdecode(file)
            fileName = os.path.join(input.decode("utf-8"), f)
            extract(fileName, extractor)


if __name__ == "__main__":
    main()
