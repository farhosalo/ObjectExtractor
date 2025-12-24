import cv2
import torch
import os

from transformers import OwlViTProcessor, OwlViTForObjectDetection
from .AbstractObjectExtractor import AbstractObjectExtractor


class ObjectExtractorOwlVit(AbstractObjectExtractor):
    __SCORE_THRESHOLD = 0.15  # keep detections with score >= this
    __TextLabels = [["traffic sign"]]

    def __init__(self, device="cpu", outputPath="OwlVitSigns", classes=[str]):
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

        # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        # results = processor.post_process_object_detection(
        #     outputs=outputs, target_sizes=target_sizes, threshold=0.1, text_labels=text_labels
        # )

        # for older transformers versions (e.g., <= 4.39)
        results = self.__Processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=self.__SCORE_THRESHOLD
        )
        # set number of text labels for each image
        # add text labels manually (same order as TEXT_PROMPTS)
        for r in results:
            r["text_labels"] = [self._ClassNames[int(i)] for i in r["labels"].tolist()]

        # Retrieve predictions for the first image for the corresponding text queries
        result = results[0]
        boxes, scores, text_labels = (
            result["boxes"],
            result["scores"],
            result["text_labels"],
        )

        for box, score, text_label in zip(boxes, scores, text_labels):
            box = [round(i, 2) for i in box.tolist()]
            if (box[3] - box[1] > self._MinimumSignSize[0]) and (
                box[2] - box[0] > self._MinimumSignSize[1]
            ):

                x1, y1, x2, y2 = [int(v) for v in box]
                self._saveExtractedObject(frame[y1:y2, x1:x2])
