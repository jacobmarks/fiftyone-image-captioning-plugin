"""Image Captioning plugin.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from importlib.util import find_spec
import os
from PIL import Image

import fiftyone as fo
import fiftyone.core.utils as fou
import fiftyone.operators as foo
from fiftyone.operators import types


transformers = fou.lazy_import("transformers")
replicate = fou.lazy_import("replicate")


def allows_replicate_models():
    """Returns whether the current environment allows replicate models."""
    return (
        find_spec("replicate") is not None
        and "REPLICATE_API_TOKEN" in os.environ
    )


def allows_hf_models():
    """
    Returns whether the current environment allows hugging face transformer
    models.
    """
    return find_spec("transformers") is not None


def get_filepath(sample):
    return (
        sample.local_path if hasattr(sample, "local_path") else sample.filepath
    )


def _execution_mode(ctx, inputs):
    delegate = ctx.params.get("delegate", False)

    if delegate:
        description = "Uncheck this box to execute the operation immediately"
    else:
        description = "Check this box to delegate execution of this task"

    inputs.bool(
        "delegate",
        default=False,
        required=True,
        label="Delegate execution?",
        description=description,
        view=types.CheckboxView(),
    )

    if delegate:
        inputs.view(
            "notice",
            types.Notice(
                label=(
                    "You've chosen delegated execution. Note that you must "
                    "have a delegated operation service running in order for "
                    "this task to be processed. See "
                    "https://docs.voxel51.com/plugins/index.html#operators "
                    "for more information"
                )
            ),
        )


# def _get_target_view(ctx, target):
#     if target == "SELECTED_SAMPLES":
#         return ctx.view.select(ctx.selected)

#     if target == "DATASET":
#         return ctx.dataset

#     return ctx.view


# def _list_target_views(ctx, inputs):
#     has_view = ctx.view != ctx.dataset.view()
#     has_selected = bool(ctx.selected)
#     default_target = "DATASET"
#     if has_view or has_selected:
#         target_choices = types.RadioGroup()
#         target_choices.add_choice(
#             "DATASET",
#             label="Entire dataset",
#             description="Run model on the entire dataset",
#         )

#         if has_view:
#             target_choices.add_choice(
#                 "CURRENT_VIEW",
#                 label="Current view",
#                 description="Run model on the current view",
#             )
#             default_target = "CURRENT_VIEW"

#         if has_selected:
#             target_choices.add_choice(
#                 "SELECTED_SAMPLES",
#                 label="Selected samples",
#                 description="Run model on the selected samples",
#             )
#             default_target = "SELECTED_SAMPLES"

#         inputs.enum(
#             "target",
#             target_choices.values(),
#             default=default_target,
#             view=target_choices,
#         )
#     else:
#         ctx.params["target"] = "DATASET"


def run_qwen_vl_chat(sample):
    filepath = get_filepath(sample)
    return replicate.run(
        "lucataco/qwen-vl-chat:50881b153b4d5f72b3db697e2bbad23bb1277ab741c5b52d80cd6ee17ea660e9",
        input={
            "image": open(filepath, "rb"),
            "prompt": "Describe the image in detail.",
        },
    )


def run_blip2(sample):
    filepath = get_filepath(sample)
    return replicate.run(
        "andreasjansson/blip-2:4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608",
        input={
            "image": open(filepath, "rb"),
            "question": "Describe the image?",
        },
    )


def run_fuyu8b(sample):
    filepath = get_filepath(sample)
    return replicate.run(
        "lucataco/fuyu-8b:42f23bc876570a46f5a90737086fbc4c3f79dd11753a28eaa39544dd391815e9",
        input={"image": open(filepath, "rb"), "prompt": "Describe the image."},
    )


def run_llava13b(sample):
    filepath = get_filepath(sample)
    response = replicate.run(
        "yorickvp/llava-13b:2facb4a474a0462c15041b78b1ad70952ea46b5ec6ad29583c0b29dbd4249591",
        input={"image": open(filepath, "rb"), "prompt": "Describe the image."},
    )

    resp_string = ""
    for r in response:
        resp_string += r
    return resp_string


HF_I2T_MODELS = (
    "microsoft/git-base",
    "Salesforce/blip-image-captioning-base",
    "llava-hf/llava-1.5-7b-hf",
    "nlpconnect/vit-gpt2-image-captioning",
)


REPLCATE_MODELS = {
    "blip2": run_blip2,
    "fuyu-8b": run_fuyu8b,
    "llava-13b": run_llava13b,
    "qwen-vl-chat": run_qwen_vl_chat,
}


def run_hf_model(sample, model_name):
    from transformers import pipeline

    pipe = pipeline("image-to-text", model=model_name)
    image = Image.open(get_filepath(sample))
    res = pipe(image, max_new_tokens=100)
    if type(res) == list:
        res = res[0]
    return res["generated_text"]


def generate_sample_caption(sample, model_name):
    if model_name in HF_I2T_MODELS:
        return run_hf_model(sample, model_name)
    else:
        return REPLCATE_MODELS[model_name](sample)


class CaptionImages(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="caption_images",
            label="Caption Images",
            dynamic=True,
            execute_as_generator=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Image Captioning",
            description="Generate captions for images with state-of-the-art models",
        )

        rep_flag = allows_replicate_models()
        hf_flag = allows_hf_models()
        if not rep_flag and not hf_flag:
            inputs.message(
                "message",
                label="No models available. Please set up your environment variables or install `transformers`.",
            )
            return types.Property(inputs)

        available_models = []
        if rep_flag:
            available_models.extend(REPLCATE_MODELS.keys())
        if hf_flag:
            available_models.extend(HF_I2T_MODELS)

        available_models.sort()

        model_choices = types.RadioGroup()
        for model in available_models:
            model_choices.add_choice(model, label=model)

        inputs.enum(
            "model_name",
            model_choices.values(),
            label="Model",
            view=types.DropdownView(),
            required=True,
        )

        inputs.str(
            "caption_field",
            label="Caption field",
            description="The name of the field to store the generated captions",
            required=True,
        )

        inputs.view_target(ctx)
        _execution_mode(ctx, inputs)
        return types.Property(inputs, view=form_view)

    async def execute(self, ctx):
        sample_collection = ctx.target_view()
        model_name = ctx.params["model_name"]
        caption_field = ctx.params["caption_field"]

        ctx.dataset.add_sample_field(caption_field, ftype=fo.StringField)

        num_samples = sample_collection.count()
        captions = []

        for i, sample in enumerate(
            sample_collection.iter_samples(progress=True, autosave=True)
        ):
            captions.append(generate_sample_caption(sample, model_name))

            progress_label = f"Loading {i} of {num_samples}"
            progress_view = types.ProgressView(label=progress_label)
            loading_schema = types.Object()
            loading_schema.int("percent_complete", view=progress_view)
            show_output_params = {
                "outputs": types.Property(loading_schema).to_json(),
                "results": {"percent_complete": i / num_samples},
            }
            yield ctx.trigger("show_output", show_output_params)

        sample_collection.set_values(caption_field, captions)
        sample_collection.save()
        yield ctx.trigger("reload_dataset")

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)


def register(plugin):
    plugin.register(CaptionImages)
