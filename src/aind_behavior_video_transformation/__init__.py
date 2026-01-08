"""Init package"""

__version__ = "0.2.2"

from aind_behavior_video_transformation.etl import (  # noqa F401
    BehaviorVideoJob,
    BehaviorVideoJobSettings,
)
from aind_behavior_video_transformation.transform_videos import (  # noqa F401
    CompressionEnum,
    CompressionRequest,
)
