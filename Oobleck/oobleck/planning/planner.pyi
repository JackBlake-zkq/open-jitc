from cornstarch.pipeline_template import PipelineTemplate

from oobleck.planning.profiler import LayerExecutionResult

def create_pipeline_templates(
    model_name: str,
    profile_data: list[LayerExecutionResult],
    num_nodes: list[int],
) -> dict[int, PipelineTemplate]: ...
