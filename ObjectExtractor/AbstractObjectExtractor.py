import os
from abc import ABC, abstractmethod
import re
import cv2
import logging
from datetime import datetime


class AbstractObjectExtractor(ABC):
    _Model = None
    _MinimalExtractedImagesSize = (44, 44)

    def __init__(self, device, outputPath, classes=[str]):
        self.__OutputPath = outputPath
        self._Device = device
        self._ClassNames = classes

        if not os.path.exists(self.__OutputPath):
            os.makedirs(self.__OutputPath)

        self.__FileIndex = 0

    def setMinimumObjectSize(self, width, height):
        self._MinimalExtractedImagesSize = (width, height)

    def extractFromImage(self, imagePath):
        frame = cv2.imread(imagePath)
        self._extractObjectsFromFrame(frame)

    def extractFromVideo(self, VideoFile):
        videoCapture = cv2.VideoCapture(VideoFile)
        videoFPS = videoCapture.get(cv2.CAP_PROP_FPS)
        frameCount = 0
        imagesPerSeconds = 1
        read = True
        while read:
            read, image = videoCapture.read()

            if not read:
                break

            if frameCount % int(videoFPS / imagesPerSeconds) == 0:
                self._extractObjectsFromFrame(image)
            frameCount += 1

        videoCapture.release()
        logging.info("Processing " + VideoFile + " completed.")

    @abstractmethod
    def _extractObjectsFromFrame(self):
        pass

    def _saveExtractedObject(self, object):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cv2.imwrite(os.path.join(self.__OutputPath, f"{timestamp}.jpg"), object)
            self.__FileIndex += 1
        except:
            pass
