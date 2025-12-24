import argparse
import filetype
from enum import Enum
import os

from ObjectExtractor.ObjectExtractorGDino import ObjectExtractorGDino
from ObjectExtractor.ObjectExtractorOwlVit import ObjectExtractorOwlVit
from ObjectExtractor import AbstractObjectExtractor


class SupportedFileTypes(Enum):
    UNKNOWN = 0
    VIDEO = 1
    IMAGE = 2


def GetFileType(filename: str):
    mime = filetype.guess_mime(filename)
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
        print("Input file format is not supported")


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

    extractor = ObjectExtractorOwlVit(
        device=args.device, outputPath="ObjectExtractorOutput", classes=args.classes
    )
    extractor.setMinimumObjectSize(24, 24)

    input = os.fsencode(args.input)
    if not os.path.exists(input):
        print("Input path does not exist")
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
