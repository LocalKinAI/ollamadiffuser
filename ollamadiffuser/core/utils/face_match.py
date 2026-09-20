"""Is it still the same person? Faces compared by embedding, not by eye.

An identity-preserving pipeline — edit a portrait into a scene, film the scene,
carry on from the film's last frame — needs to know when the identity was not
preserved, and a vision-language model is the wrong witness: shown a profile
that turned to the lens as somebody else, it reported that nothing had changed.
A face-recognition embedding gives a number instead.

The models are OpenCV Zoo's, because OpenCV can run them with nothing else
installed (``cv2.FaceDetectorYN`` / ``cv2.FaceRecognizerSF``, in opencv-python
since 4.5.4) and because of what they are licensed for:

- ``face_detection_yunet_2023mar.onnx`` — YuNet, MIT, 0.2 MB
- ``face_recognition_sface_2021dec.onnx`` — SFace, Apache-2.0, 37 MB

The better-known ArcFace weights that ship with insightface are for
non-commercial research only, which rules them out of anything meant to be
given to other people.

Nothing is downloaded by importing or by serving: the two files are fetched by
``python -m ollamadiffuser.core.utils.face_match --download`` (or put in place
by hand) under ``<models_dir>/face/``.
"""
from __future__ import annotations

import logging
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from ..config.settings import settings

logger = logging.getLogger(__name__)

DETECTOR = "face_detection_yunet_2023mar.onnx"
RECOGNIZER = "face_recognition_sface_2021dec.onnx"
_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
SOURCES = {
    DETECTOR: f"{_ZOO}/face_detection_yunet/{DETECTOR}",
    RECOGNIZER: f"{_ZOO}/face_recognition_sface/{RECOGNIZER}",
}
# OpenCV's own figure for SFace with cosine distance: at or above it, the same
# person. Generated faces score lower than photographs of a real one, so a
# caller judging generated frames will want its own, measured on its own.
SAME_PERSON = 0.363


class FaceModelsMissing(RuntimeError):
    """The two model files are not where they are looked for."""


def models_dir() -> Path:
    return Path(settings.models_dir) / "face"


@dataclass
class Found:
    """The largest face in a picture: where it is and what it looks like."""
    box: List[int]            # x, y, width, height, in the picture's pixels
    feature: np.ndarray


class FaceMatcher:
    def __init__(self, directory: Optional[Path] = None):
        self.directory = Path(directory) if directory else models_dir()
        self._detector = None
        self._recognizer = None

    def _load(self):
        if self._detector is not None:
            return
        missing = [name for name in (DETECTOR, RECOGNIZER) if not (self.directory / name).exists()]
        if missing:
            raise FaceModelsMissing(
                f"face models not found in {self.directory}: {', '.join(missing)}. "
                f"Fetch them with `python -m ollamadiffuser.core.utils.face_match --download` "
                f"(OpenCV Zoo, 37 MB), or put the files there.")
        import cv2
        self._detector = cv2.FaceDetectorYN.create(str(self.directory / DETECTOR), "", (320, 320), 0.6, 0.3, 50)
        self._recognizer = cv2.FaceRecognizerSF.create(str(self.directory / RECOGNIZER), "")

    def find(self, picture: Image.Image) -> Optional[Found]:
        """The largest face in the picture, or None when there is none to see —
        she has her back to the camera, or is too small to be anybody."""
        self._load()
        import cv2
        image = cv2.cvtColor(np.asarray(picture.convert("RGB")), cv2.COLOR_RGB2BGR)
        self._detector.setInputSize((image.shape[1], image.shape[0]))
        _, faces = self._detector.detect(image)
        if faces is None or len(faces) == 0:
            return None
        face = max(faces, key=lambda row: row[2] * row[3])
        aligned = self._recognizer.alignCrop(image, face)
        return Found(box=[int(v) for v in face[:4]], feature=self._recognizer.feature(aligned))

    def alike(self, one: Found, other: Found) -> float:
        """Cosine similarity of two faces: 1 is the same picture, 0 is strangers."""
        self._load()
        import cv2
        return float(self._recognizer.match(one.feature, other.feature, cv2.FaceRecognizerSF_FR_COSINE))

    def compare(self, anchor: Image.Image, pictures: List[Image.Image]) -> dict:
        """Each picture's largest face against the anchor's.

        A picture with no face in it is reported as such rather than scored:
        "no face" and "a different face" call for different cures.
        """
        who = self.find(anchor)
        if who is None:
            raise ValueError("no face found in the anchor picture")
        results = []
        for picture in pictures:
            found = self.find(picture)
            if found is None:
                results.append({"found": False})
            else:
                results.append({"found": True, "box": found.box, "width": found.box[2],
                                "similarity": round(self.alike(who, found), 4)})
        return {"model": "sface", "same_person_at": SAME_PERSON, "anchor": {"box": who.box}, "results": results}


def download(directory: Optional[Path] = None) -> Path:
    """Fetch the two model files from the OpenCV Zoo. Only ever run on purpose."""
    target = Path(directory) if directory else models_dir()
    target.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        path = target / name
        if path.exists() and path.stat().st_size > 100_000:
            print(f"have  {path}")
            continue
        print(f"fetch {url}")
        partial = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, partial)
        partial.replace(path)
        print(f"wrote {path} ({path.stat().st_size // 1024} KB)")
    return target


if __name__ == "__main__":
    if "--download" in sys.argv:
        download()
    else:
        print(__doc__)
