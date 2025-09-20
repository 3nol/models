"""Video vit linear evaluation task definition."""

from official import vision
from official.core import task_factory

from official.projects.tubevit.configs import tubevit as tubevit_cfg


@task_factory.register_task_cls(tubevit_cfg.ViTLinearEvalTaskConfig)
class VideoViTEvalTask(vision.VideoClassificationTask):
    """A task for video vit linear evaluation."""

    pass
