Place the NoiseControl LoRA weights anywhere and pass them with `--lora_path`, e.g.
`--lora_path checkpoints/pytorch_lora_weights.safetensors`.

Without `--lora_path`, `infer.py` / `infer_from_flow.py` look for the default training output path under `ckpt/`;
if the file is missing they print a warning and run the base CogVideoX model without the LoRA.
