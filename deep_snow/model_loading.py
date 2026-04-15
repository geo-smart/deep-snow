import torch

from deep_snow.errors import ModelCompatibilityError


def load_resdepth_checkpoint(model, model_path, gpu):
    try:
        if gpu:
            state_dict = torch.load(model_path, weights_only=True)
            model.load_state_dict(state_dict)
            model.to("cuda")
        else:
            state_dict = torch.load(
                model_path,
                map_location=torch.device("cpu"),
                weights_only=True,
            )
            model.load_state_dict(state_dict)
    except RuntimeError as exc:
        if "size mismatch" not in str(exc):
            raise

        from deep_snow.prediction import get_prediction_input_channels

        expected_inputs = model.encoder[0][0][0].weight.shape[1]
        raise ModelCompatibilityError(
            "The checkpoint at "
            f"'{model_path}' is incompatible with the current deep-snow input feature set. "
            f"The current pipeline expects {expected_inputs} input channels "
            f"({', '.join(get_prediction_input_channels())}). "
            "Use the current default model or ensemble from the GitHub Actions workflow, "
            "or choose a checkpoint trained with the same feature stack."
        ) from exc
