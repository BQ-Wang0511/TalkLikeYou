# coding: utf-8
# pylint: disable=wrong-import-position
"""InsightFace: A Face Analysis Toolkit."""
from __future__ import absolute_import

try:
    import onnxruntime
except ImportError:
    raise ImportError(
        "Unable to import dependency onnxruntime. "
    )

__version__ = '0.7.3'

from . import model_zoo
from . import app
