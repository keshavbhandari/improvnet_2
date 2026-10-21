"""Compatibility entrypoint; canonical melody inference now batches ready tasks."""
from improvnet.inference.melody_inference import main
# Keep scheduler helpers importable for existing callers.
from improvnet.inference.melody_pipeline import ForwardRequest, forward_batch, run_pipeline

if __name__ == '__main__':
    main(pipeline_runner=run_pipeline)
