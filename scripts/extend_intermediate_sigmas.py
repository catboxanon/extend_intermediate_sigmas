import logging
import gradio as gr
from modules import script_callbacks, scripts
import types
from typing import Any
from functools import partial
import sys
import traceback
import torch
import math


class ExtendIntermediateSigmas:
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required":
                    {"sigmas": ("SIGMAS", ),
                     "steps": ("INT", {"default": 2, "min": 1, "max": 100}),
                     "start_at_sigma": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 20000.0, "step": 0.01, "round": False}),
                     "end_at_sigma": ("FLOAT", {"default": 12.0, "min":  0.0, "max": 20000.0, "step": 0.01, "round": False}),
                     "spacing": (['linear', 'cosine', 'sine'],),
                    }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    FUNCTION = "extend"

    def extend(self, sigmas: torch.Tensor, steps: int, start_at_sigma: float, end_at_sigma: float, spacing: str):
        if start_at_sigma < 0:
            start_at_sigma = float("inf")

        interpolator = {
            'linear': lambda x: x,
            'cosine': lambda x: torch.sin(x*math.pi/2),
            'sine':   lambda x: 1 - torch.cos(x*math.pi/2)
        }[spacing]

        # linear space for our interpolation function
        x = torch.linspace(0, 1, steps + 1, device=sigmas.device)[1:-1]
        computed_spacing = interpolator(x)

        extended_sigmas = []
        for i in range(len(sigmas) - 1):
            sigma_current = sigmas[i]
            sigma_next = sigmas[i+1]

            extended_sigmas.append(sigma_current)

            if end_at_sigma <= sigma_current <= start_at_sigma:
                interpolated_steps = computed_spacing * (sigma_next - sigma_current) + sigma_current
                extended_sigmas.extend(interpolated_steps.tolist())

        # Add the last sigma value
        if len(sigmas) > 0:
            extended_sigmas.append(sigmas[-1])

        extended_sigmas = torch.FloatTensor(extended_sigmas)

        return (extended_sigmas,)


class ExtendIntermediateSigmasScript(scripts.Script):
    def __init__(self):
        self.extend_intermediate_sigmas_enabled = False
        self.steps = 2
        self.start_at_sigma = -1.0
        self.end_at_sigma = 12.0
        self.spacing = 'linear'

    sorting_priority = 20

    def title(self):
        return "Extend intermediate sigmas"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            with gr.Row():
                self.extend_intermediate_sigmas_enabled = gr.Checkbox(label="Enable", value=False)
                self.steps = gr.Slider(label="Steps", value=2, minimum=1, maximum=100, step=1)
                self.start_at_sigma = gr.Number(label="Start at sigma", value=-1.0, precision=2, step=0.01)
                self.end_at_sigma = gr.Number(label="End at sigma", value=12.0, precision=2, step=0.01)
                self.spacing = gr.Dropdown(label="Spacing", choices=['linear', 'cosine', 'sine'], value='linear')

        return (self.extend_intermediate_sigmas_enabled, self.steps, self.start_at_sigma, self.end_at_sigma, self.spacing)

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 4:  # Updated to account for the new parameter
            (self.extend_intermediate_sigmas_enabled, self.steps, self.start_at_sigma, self.end_at_sigma, self.spacing) = args[:5]
        else:
            logging.warning("Not enough arguments provided to process_before_every_sampling")
            return

        xyz = getattr(p, "_extend_intermediate_sigmas_xyz", {})
        if "extend_intermediate_sigmas_enabled" in xyz:
            self.modify_first_sigma_enabled = xyz["modify_first_sigma_enabled"] == "True"
        if "steps" in xyz:
            self.steps = int(xyz["steps"])
        if "start_at_sigma" in xyz:
            self.start_at_sigma = float(xyz["start_at_sigma"])
        if "end_at_sigma" in xyz:
            self.end_at_sigma = float(xyz["end_at_sigma"])
        if "spacing" in xyz:
            self.spacing = xyz["spacing"]

        if not self.extend_intermediate_sigmas_enabled:
            return

        # Get the original sigmas
        if hasattr(p.sampler, 'get_sigmas'):
            original_sigmas = p.sampler.get_sigmas(p, p.steps)
        elif hasattr(p.sampler, 'model_wrap'):
            original_sigmas = p.sampler.model_wrap.sigmas
        else:
            logging.warning("Unable to access sigmas from the sampler")
            return
        sigmas = ExtendIntermediateSigmas().extend(original_sigmas, self.steps, self.start_at_sigma, self.end_at_sigma, 'linear')[0]

        # Apply the modified sigmas
        if hasattr(p.sampler, 'model_wrap'):
            p.sampler.model_wrap.sigmas = sigmas
            p.sampler.model_wrap.log_sigmas = sigmas.log()

        # Override the get_sigmas method if it exists
        if hasattr(p.sampler, 'get_sigmas'):
            def new_get_sigmas(self, p, steps):
                return sigmas
            p.sampler.get_sigmas = types.MethodType(new_get_sigmas, p.sampler)

        p.extra_generation_params.update({
            "extend_intermediate_sigmas_enabled": self.extend_intermediate_sigmas_enabled,
            "steps": self.steps,
            "start_at_sigma": self.start_at_sigma,
            "end_at_sigma": self.end_at_sigma,
            "spacing": self.spacing,
        })

        return

def set_value(p, x: Any, xs: Any, *, field: str):
    if not hasattr(p, "_extend_intermediate_sigmas_xyz"):
        p._extend_intermediate_sigmas_xyz = {}
    p._extend_intermediate_sigmas_xyz[field] = x

def make_axis_on_xyz_grid():
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break

    if xyz_grid is None:
        return

    axis = [
        xyz_grid.AxisOption(
            "(Extend intermediate sigmas) Enable",
            str,
            partial(set_value, field="extend_intermediate_sigmas_enabled"),
            choices=lambda: ["True", "False"]
        ),
        xyz_grid.AxisOption(
            "(Extend intermediate sigmas) Steps",
            int,
            partial(set_value, field="steps"),
        ),
        xyz_grid.AxisOption(
            "(Extend intermediate sigmas) Start at sigma",
            float,
            partial(set_value, field="start_at_sigma"),
        ),
        xyz_grid.AxisOption(
            "(Extend intermediate sigmas) End at sigma",
            float,
            partial(set_value, field="end_at_sigma"),
        ),
        xyz_grid.AxisOption(
            "(Extend intermediate sigmas) Spacing",
            str,
            partial(set_value, field="spacing"),
            choices=lambda: ['linear', 'cosine', 'sine']
        ),
    ]

    if not any(x.label.startswith("(Extend intermediate sigmas)") for x in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(axis)

def on_before_ui():
    try:
        make_axis_on_xyz_grid()
    except Exception:
        error = traceback.format_exc()
        print(
            f"[-] Sigma test: xyz_grid error:\n{error}",
            file=sys.stderr,
        )

script_callbacks.on_before_ui(on_before_ui)
