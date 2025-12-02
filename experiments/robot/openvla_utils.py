"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")
    print("[*] Low CPU Memory Usage Enabled 300")

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    base_ckpt_path = cfg.pretrained_checkpoint
    adapter_path = Path(cfg.adapter_path) if getattr(cfg, "adapter_path", None) else None
    
    # If the adapter path already contains a merged model (config + weights), prefer loading from it
    model_load_path = adapter_path if adapter_path is not None and (adapter_path / "config.json").exists() else base_ckpt_path
    print(adapter_path, model_load_path)
    vla = AutoModelForVision2Seq.from_pretrained(
        base_ckpt_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    
    # Optionally load and merge a LoRA adapter (adapter_path points to saved adapter or merged run dir)
    if adapter_path is not None and adapter_path.exists():
        adapter_config = adapter_path / "adapter_config.json"
        if adapter_config.exists():
            vla = PeftModel.from_pretrained(vla, adapter_path)
            #print(vla)
            vla = vla.merge_and_unload()
        else:
            # If no adapter config, assume weights are already merged in the adapter_path directory
            pass

    # Move model to device.
    # Note: `.to()` is not supported for 8-bit or 4-bit bitsandbytes models, but the model will
    #       already be set to the right devices and casted to the correct dtype upon loading.
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # Load dataset stats used during finetuning (for action un-normalization).
    adapter_path = Path(base_ckpt_path) if adapter_path is None else adapter_path
    stats_candidates = []
    if adapter_path is not None:
        stats_candidates.append(adapter_path / "dataset_statistics.json")
    #stats_candidates.append(Path(base_ckpt_path) / "dataset_statistics.json")
    stats_loaded = False
    for p in stats_candidates:
        if p.is_file():
            with open(p, "r") as f:
                print(f"[*] Loading dataset statistics from {str(p)} 4000")
                stats = json.load(f)
                vla.norm_stats = stats
                vla.base_model.norm_stats = stats  
                vla.config.dataset_statistics = stats
                vla.base_model.config.dataset_statistics = stats
                #vla.model.dataset_statistics = stats
                #vla.base_model.model.dataset_statistics = stats
                #vla.base_model.model.norm_stats = stats
            stats_loaded = True
            break
    print(vla.norm_stats.keys() if stats_loaded else "No dataset statistics loaded.")
    if not stats_loaded:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )
    def print_all_stats(vla):
        print("\n================= ALL STATS IN MODEL =================\n")

        # 1 — top-level PEFT wrapper
        print("vla.norm_stats:", getattr(vla, "norm_stats", None))

        # 2 — base model
        if hasattr(vla, "base_model"):
            print("\n[base_model]")
            print("base_model.norm_stats:", getattr(vla.base_model, "norm_stats", None))

        # 3 — nested model (OpenVLAForActionPrediction)
        if hasattr(vla, "model"):
            print("\n[self.model]")
            print("self.model.dataset_statistics:", getattr(vla.model, "dataset_statistics", None))
            print("self.model.norm_stats:", getattr(vla.model, "norm_stats", None))

        if hasattr(vla, "base_model") and hasattr(vla.base_model, "model"):
            print("\n[base_model.model]")
            print("base_model.model.dataset_statistics:", getattr(vla.base_model.model, "dataset_statistics", None))
            print("base_model.model.norm_stats:", getattr(vla.base_model.model, "norm_stats", None))

        # 4 — search every submodule recursively
        print("\n[Recursive search for anything named 'stats']")

        for name, module in vla.named_modules():
            for attr in ["norm_stats", "dataset_statistics", "stats"]:
                if hasattr(module, attr):
                    value = getattr(module, attr)
                    if isinstance(value, dict):
                        print(f"\nFound in module: {name}.{attr}")
                        print("Keys:", list(value.keys()))

        print("\n================= END ALL STATS =================\n")
        
    # print_all_stats(vla)


    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    candidates = []
    if getattr(cfg, "adapter_path", None):
        candidates.append(Path(cfg.adapter_path))
    candidates.append(Path(cfg.pretrained_checkpoint))

    last_error = None
    for root in candidates:
        try:
            return AutoProcessor.from_pretrained(root, trust_remote_code=True)
        except OSError as e:
            last_error = e
            continue

    # If we made it here, re-raise the last error for visibility
    raise last_error


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process inputs.
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
    
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action
