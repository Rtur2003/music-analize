"""Constants used throughout the application."""

from __future__ import annotations

# Audio processing constants
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_DURATION_SEC = 30
DEFAULT_LUFS_TARGET = -23.0
MIN_AUDIO_DURATION_SEC = 0.1
MAX_AUDIO_DURATION_SEC = 600
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 192000
DEFAULT_FRAME_MS = 50

# Feature extraction constants
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_N_MELS = 128
DEFAULT_N_MFCC = 20
DEFAULT_FMIN = 20
DEFAULT_FMAX = 20000
EPSILON = 1e-9

# Model constants
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.5
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

# File size limits (bytes)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_TEMP_FILE_AGE_SECONDS = 3600  # 1 hour

# Supported file extensions
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

# Model file names
GENRE_MODEL_FILENAME = "genre_classifier.joblib"
AUTH_MODEL_FILENAME = "auth_classifier.joblib"
LABEL_ENCODER_FILENAME = "genre_label_encoder.joblib"

# Report settings
CHROMA_LABELS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Validation
MIN_SAMPLES_PER_CLASS = 2
MIN_FEATURE_COUNT = 1
