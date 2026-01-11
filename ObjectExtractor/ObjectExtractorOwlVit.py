import torch

from transformers import OwlViTProcessor, OwlViTForObjectDetection
from .AbstractObjectExtractor import AbstractObjectExtractor


class ObjectExtractorOwlVit(AbstractObjectExtractor):
    __SCORE_THRESHOLD = 0.15  # keep detections with score >= this

    def __init__(self, device="cpu", outputPath="OwlVitOut", classes=[str]):
        super().__init__(device, outputPath, classes)
        self.__DownloadModel()

    def __DownloadModel(self):
        self.__Processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self._Model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-base-patch32"
        ).to(self._Device)

    def _extractObjectsFromFrame(self, frame):
        inputs = self.__Processor(
            text=self._ClassNames, images=frame, return_tensors="pt"
        ).to(self._Device)
        outputs = self._Model(**inputs)

        height, width = frame.shape[:2]
        target_sizes = torch.tensor([(height, width)])

        results = self.__Processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=self.__SCORE_THRESHOLD
        )

        for r in results:
            r["text_labels"] = [self._ClassNames[int(i)] for i in r["labels"].tolist()]

            for box in results[0]["boxes"]:
                if not all(x > 0 for x in box):
                    continue
                box = [round(i, 2) for i in box.tolist()]
                x1, y1, x2, y2 = [int(v) for v in box]

                if (x2 - x1 > self._MinimalExtractedImagesSize[0]) and (
                    y2 - y1 > self._MinimalExtractedImagesSize[1]
                ):
                    self._saveExtractedObject(frame[y1:y2, x1:x2])
