"""
SpeechLLM: 語音大語言模型框架
基於 Qwen-2/2.5 和 Whisper encoder 的多模態語音對話系統
"""

__version__ = "0.1.0"
__author__ = "SpeechLLM Team"

from .models import *
from .codecs import *
from .training import *
from .inference import *
