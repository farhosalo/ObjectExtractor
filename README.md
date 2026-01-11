# Object extractor

This application aims to extract objects from videos or images inputs.

Features:

- Processes input from videos or images
- Detects passed objects and saves them to a directory

Technologies used:
The user can choose between one of the following underlying technologies:

- **[OWL-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit)**
- **[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)**

## Prerequisites

- **Python** version >= 3.11
- All required Python packages are listed in the `requirements.txt` file.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/farhosalo/ObjectExtractor.git
   cd ObjectExtractor
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Configuration (For Expert Users)

Please don't modifying the configuration unless you have extensive experience in creating such models.

The configuration is stored in a Python file named Configuration.py located in the project root directory.

- **“OUTPUT_PATH“:** The path to the directory where the extracted images will be saved.
- **"MINIMUM_HEIGHT":** Minimum height of the object to be extracted
- **"MINIMUM_WIDTH":** Minimum width of the object to be extracted
- **"UNDERLYING_MODEL":** Underlying model to use for object extraction. Supported models: "OwlVit", "GroundingDINO"

## Usage

To use the project, run the `main.py`.
Options:

- -h, --help            show help message and exit

- -d {cuda,mps,cpu}, --device {cuda,mps,cpu} Selects the accelerating device, default is cpu, cuda for NVIDIA GPU and mps for Apple GPU
- -i INPUT, --input INPUT Path to the input image, video or directory that contains videos ans images
- -c CLASSES [CLASSES ...], --classes CLASSES [CLASSES ...] Space separated list of classes to extract (e.g. "traffic sign" "bus")

```bash
python main.py --device mps --input <path/to/video.mp4> --classes "traffic sign" "bus"
```

This project does not include pre-trained model weights. At runtime, the following models are automatically downloaded from their official sources:

1. GroundingDINO

   - Project: GroundingDINO (IDEA-Research)
   - Repository: <https://github.com/IDEA-Research/GroundingDINO>
   - Code License: Apache License 2.0
   - Model Weights: Downloaded from official GitHub releases

   ⚠️ The GroundingDINO repository doesn’t explicitly mention a license for the released model weight files (.pth). Consequently, the weights won’t be **redistributed** with this project. Users are responsible for reviewing and adhering to the original terms, particularly for commercial use.

2. OWL-ViT (google/owlvit-base-patch32)
   - Provider: Google (via Hugging Face)
   - Model Page: <https://huggingface.co/google/owlvit-base-patch32>
   - License: Apache License 2.0

   The Hugging Face transformers library automatically downloads the model.

## Contributing

Contributions are welcome! If you find any bugs or have ideas for new features, feel free to open an issue or submit a pull request.

## License

- This project is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). See the [LICENSE](LICENSE) file for details.

- Additional copyright and attribution notices can be found in the [NOTICE](NOTICE.md) file.

- Third-Party Libraries and Licenses: This project uses several third-party libraries. See [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES.md) for a complete list of dependencies and their licenses.

## Disclaimer

- This repository exclusively contains original source code.
- Pretrained model weights are downloaded directly from official sources.
- Users are responsible for ensuring that they comply with the license terms when using third-party models.
