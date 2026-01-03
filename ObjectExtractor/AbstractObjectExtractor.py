import logging.config
import os
from abc import ABC, abstractmethod
import re
import cv2
import logging

logging.basicConfig(level=logging.INFO)


class AbstractObjectExtractor(ABC):
    _Model = None
    _MinimumSignSize = (44, 44)

    def __init__(self, device, outputPath, classes=[str]):
        self.__OutputPath = outputPath
        self._Device = device
        self._ClassNames = classes

        if not os.path.exists(self.__OutputPath):
            os.makedirs(self.__OutputPath)

        self.__FileIndex = self.__getMaxIndex() + 1

    def setMinimumObjectSize(self, width, height):
        self._MinimumSignSize = (width, height)

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

    def __getMaxIndex(self):
        def extract_number(f):
            s = re.findall("(\d+).jpg", f)
            return (int(s[0]) if s else -1, f)

        fileList = os.listdir(self.__OutputPath)

        if len(fileList) == 0:
            return -1

        maxFile = max(fileList, key=extract_number)
        if maxFile and maxFile.endswith(".jpg"):
            maxFile = os.path.splitext(maxFile)[0]
            return int(maxFile.split("_", 1)[1])
        return -1

    def _saveExtractedObject(self, object):
        try:
            fileName = "Object_%08d.jpg" % (self.__FileIndex)
            cv2.imwrite(os.path.join(self.__OutputPath, fileName), object)
            self.__FileIndex += 1
        except:
            pass
