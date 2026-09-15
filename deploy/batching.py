"""RoboCasa XR-1 collation and per-request deterministic action sampling."""
import torch
import torch.nn.functional as F

REQUIRED = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw",
            "state", "action_mask", "task_id", "seed"}


def collate(requests, pad_token_id):
    """Left-pad text; concatenate flattened image patches in request order."""
    length = max(r["input_ids"].shape[1] for r in requests)
    batch = {}
    for key in ("input_ids", "attention_mask"):
        batch[key] = torch.cat([
            F.pad(r[key], (length - r[key].shape[1], 0),
                  value=pad_token_id if key == "input_ids" else 0)
            for r in requests], dim=0)
    for key in ("pixel_values", "image_grid_thw", "state", "action_mask"):
        batch[key] = torch.cat([r[key] for r in requests], dim=0)
    return batch


def seeded_noise(action_mask, seeds):
    # Independent generators keep queue order from changing action noise.
    return torch.cat([
        torch.randn((1, *action_mask.shape[1:]), device=action_mask.device,
                    dtype=action_mask.dtype,
                    generator=torch.Generator(device=action_mask.device).manual_seed(seed))
        for seed in seeds], dim=0)


class XR1BatchPolicy:
    def __init__(self, model_path):
        from transformers import AutoModel, AutoProcessor
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, attn_implementation="flash_attention_2",
            dtype=torch.bfloat16).cuda().eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False)
        self.pad_token_id = self.processor.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.processor.tokenizer.eos_token_id
        self.action_shape = tuple(self.processor.get_action_mask("robocasa_mg").shape[1:])
        self.state_dim = self.model.config.state_dim

    def validate(self, data):
        if not isinstance(data, dict) or set(data) != REQUIRED:
            raise ValueError(f"Expected fields: {sorted(REQUIRED)}")
        if data["task_id"] != "robocasa_mg":
            raise ValueError("This batched policy serves robocasa_mg")
        if not isinstance(data["seed"], int) or not 0 <= data["seed"] < 2**63:
            raise ValueError("seed must be a nonnegative 63-bit integer")
        for key in REQUIRED - {"task_id", "seed"}:
            if not isinstance(data[key], torch.Tensor) or data[key].device.type != "cpu":
                raise ValueError(f"{key} must be a CPU tensor")
        if data["input_ids"].ndim != 2 or data["input_ids"].shape[0] != 1:
            raise ValueError("Send exactly one environment observation per request")
        if not 0 < data["input_ids"].shape[1] <= 8192:
            raise ValueError("Invalid text length")
        if data["attention_mask"].shape != data["input_ids"].shape:
            raise ValueError("attention_mask must match input_ids")
        if data["state"].shape != (1, 1, self.state_dim):
            raise ValueError("Unexpected state shape")
        if tuple(data["action_mask"].shape) != (1, *self.action_shape):
            raise ValueError("Unexpected action_mask shape")
        if data["image_grid_thw"].shape != (3, 3) or data["pixel_values"].ndim != 2:
            raise ValueError("Expected three RoboCasa camera images")

    @torch.inference_mode()
    def __call__(self, requests):
        batch = collate(requests, self.pad_token_id)
        batch = {k: v.to(device=self.model.device,
                        dtype=self.model.dtype if v.is_floating_point() else v.dtype)
                 for k, v in batch.items()}
        state, action_mask = batch.pop("state"), batch.pop("action_mask")
        model = self.model
        vlm = model.vlm(**batch, use_cache=True)
        bs, action_length, _ = action_mask.shape
        query_length = action_length + state.shape[1] + 1
        positions = (torch.arange(query_length, device=action_mask.device)
                     .view(1, 1, -1).repeat(3, bs, 1)
                     + vlm.position_ids.max(dim=-1)[0][..., None] + 1)
        position_embeds = model.rotary_emb(action_mask, positions)
        dit_mask = torch.tril(torch.ones((bs, query_length, query_length), device=action_mask.device))
        cache_mask = vlm.attention_mask[:, None, :].expand(-1, query_length, -1)
        attn_mask = torch.cat([cache_mask, dit_mask], dim=-1)[:, None].bool()
        state_embed = model.state_projector(state)
        x = seeded_noise(action_mask, [r["seed"] for r in requests])
        # Same five-step Euler sampler as the published RoboCasa checkpoint.
        for step in range(5):
            t = torch.full((bs, 1, 1), step / 5, device=x.device, dtype=x.dtype)
            x = x + model.dit_forward(
                noisy_action=x, t=t, action_mask=action_mask,
                state_embed=state_embed, position_embeds=position_embeds,
                past_key_values=vlm.past_key_values, attn_mask=attn_mask) / 5
        actions = x.cpu()
        return list(actions.split(1, dim=0))
