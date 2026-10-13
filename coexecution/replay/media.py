"""Explicit media manifest. No caller-supplied path becomes an HTTP route."""

import shutil
from pathlib import Path

from .schema import digest, source, under_root


class MediaRegistry:
    def __init__(self, output):
        self.output, self.assets = Path(output), {}

    def add(self, root, relative, *, mime):
        suffixes = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "video/webm": ".webm",
            "video/mp4": ".mp4",
        }
        if mime not in suffixes:
            raise ValueError("Unsupported replay media type")
        path = under_root(root, relative)
        provenance = source(path)
        identity = "media-" + digest(provenance)[:24]
        target = Path("media") / (identity + suffixes[mime])
        if identity not in self.assets:
            destination = self.output / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            self.assets[identity] = {
                "id": identity,
                "path": str(target),
                "mime": mime,
                "source": provenance,
            }
        return identity
