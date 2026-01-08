from groundingdino.util.inference import load_model, Model
from urllib import request

from .AbstractObjectExtractor import AbstractObjectExtractor
import os


class ObjectExtractorGDino(AbstractObjectExtractor):
    __GDinoModelUrl = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    __GDinoConfigUrl = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/refs/tags/v0.1.0-alpha2/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    __WeightDir = os.path.join(os.getcwd(), "Weights")

    def __init__(self, device="cpu", outputPath="Output", classes=list[str]):
        classNames = self.__enhanceClassName(classes)
        super().__init__(device, outputPath, classNames)
        self.__downloadModel()

    def __enhanceClassName(self, classNames: list[str]) -> list[str]:
        return [f"all {className}s" for className in classNames]

    def __downloadModel(self):
        if not os.path.exists(self.__WeightDir):
            os.makedirs(self.__WeightDir)

        for url in [self.__GDinoModelUrl, self.__GDinoConfigUrl]:
            fileName = url.rsplit("/", 1)[-1]
            completeFileName = os.path.join(self.__WeightDir, fileName)

            if not os.path.exists(completeFileName):
                request.urlretrieve(url, completeFileName)

    def __loadModel(self):
        if self._Model == None:
            configPath = os.path.join(
                self.__WeightDir, self.__GDinoConfigUrl.rsplit("/", 1)[-1]
            )
            checkPoint = os.path.join(
                self.__WeightDir, self.__GDinoModelUrl.rsplit("/", 1)[-1]
            )
            self._Model = Model(
                model_config_path=configPath,
                model_checkpoint_path=checkPoint,
                device=self._Device,
            )

    def _extractObjectsFromFrame(self, frame):
        self.__loadModel()
        detections, _ = self._Model.predict_with_caption(
            image=frame,
            caption=", ".join(self._ClassNames),
            box_threshold=0.40,
            text_threshold=0.35,
        )

        for detected in detections:
            bbox = detected[0].astype(int)
            # Sign size should be greater than __MinimumSignSize pixels
            if (bbox[3] - bbox[1] > self._MinimalExtractedImagesSize[0]) and (
                bbox[2] - bbox[0] > self._MinimalExtractedImagesSize[1]
            ):
                self._saveExtractedObject(frame[bbox[1] : bbox[3], bbox[0] : bbox[2]])
