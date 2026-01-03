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

## Contributing

Contributions are welcome! If you find any bugs or have ideas for new features, feel free to open an issue or submit a pull request.

### How to Contribute

1. Fork the repository
2. Create a new branch: `git checkout -b my-feature`
3. Make your changes and commit them: `git commit -m 'Add new feature'`
4. Push the branch: `git push origin my-feature`
5. Open a pull request

## License

This project is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
