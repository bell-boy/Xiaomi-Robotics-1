"""RoboCasa XR-1 collation and per-request deterministic action sampling."""
import torch
import time
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
    for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "state", "action_mask"):
        if key not in requests[0]:
            continue
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
    def __init__(self, model_path, profile=False):
        self.profile = profile
        self.last_metrics = {}
        from transformers import AutoModel, AutoProcessor
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, attn_implementation="flash_attention_2",
            dtype=torch.bfloat16).cuda().eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False)
        self.pad_token_id = self.processor.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.processor.tokenizer.eos_token_id
        robot_types = self.processor.list_robot_types()
        self.robot_type = "robocasa365" if "robocasa365" in robot_types else "robocasa_mg"
        self.visual_keys = ("pixel_values_videos", "video_grid_thw") if self.robot_type == "robocasa365" else ("pixel_values", "image_grid_thw")
        self.required = (REQUIRED - {"pixel_values", "image_grid_thw"}) | set(self.visual_keys)
        self.action_shape = tuple(self.processor.get_action_mask(self.robot_type).shape[1:])
        self.state_dim = self.model.config.state_dim
        self.state_shape = (1, 4 if self.robot_type == "robocasa365" else 1, self.state_dim)

    def validate(self, data):
        if not isinstance(data, dict) or set(data) != self.required:
            raise ValueError(f"Expected fields: {sorted(self.required)}")
        if data["task_id"] != self.robot_type:
            raise ValueError(f"This batched policy serves {self.robot_type}")
        if not isinstance(data["seed"], int) or not 0 <= data["seed"] < 2**63:
            raise ValueError("seed must be a nonnegative 63-bit integer")
        for key in self.required - {"task_id", "seed"}:
            if not isinstance(data[key], torch.Tensor) or data[key].device.type != "cpu":
                raise ValueError(f"{key} must be a CPU tensor")
        if data["input_ids"].ndim != 2 or data["input_ids"].shape[0] != 1:
            raise ValueError("Send exactly one environment observation per request")
        if not 0 < data["input_ids"].shape[1] <= 8192:
            raise ValueError("Invalid text length")
        if data["attention_mask"].shape != data["input_ids"].shape:
            raise ValueError("attention_mask must match input_ids")
        if tuple(data["state"].shape) != self.state_shape:
            raise ValueError("Unexpected state shape")
        if tuple(data["action_mask"].shape) != (1, *self.action_shape):
            raise ValueError("Unexpected action_mask shape")
        if data[self.visual_keys[1]].shape != (3, 3) or data[self.visual_keys[0]].ndim != 2:
            raise ValueError("Expected three RoboCasa camera images")

    @torch.inference_mode()
    def __call__(self, requests):
        started = time.monotonic()
        batch = collate(requests, self.pad_token_id)
        collated = time.monotonic()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(5)] if self.profile else None
        if events:
            events[0].record()
        batch = {k: v.to(device=self.model.device,
                        dtype=self.model.dtype if v.is_floating_point() else v.dtype)
                 for k, v in batch.items()}
        transferred = time.monotonic()
        if events:
            events[1].record()
        state, action_mask = batch.pop("state"), batch.pop("action_mask")
        model = self.model
        vlm = model.vlm(**batch, use_cache=True)
        if events:
            events[2].record()
        bs, action_length, _ = action_mask.shape
        query_length = action_length + state.shape[1] + 1
        positions = (torch.arange(query_length, device=action_mask.device)
                     .view(1, 1, -1).repeat(3, bs, 1)
                     + vlm.position_ids.max(dim=-1)[0][..., None] + 1)
        positions[:, :, -action_length:] += int(getattr(model.config, "inference_action_position_offset", 0))
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
        if events:
            events[3].record()
        actions = x.cpu()
        if events:
            events[4].record()
            events[4].synchronize()
            self.last_metrics = dict(
                collate_ms=(collated-started)*1000,
                h2d_wall_ms=(transferred-collated)*1000,
                h2d_stream_ms=events[0].elapsed_time(events[1]),
                vlm_stream_ms=events[1].elapsed_time(events[2]),
                action_head_stream_ms=events[2].elapsed_time(events[3]),
                d2h_stream_ms=events[3].elapsed_time(events[4]))
        return list(actions.split(1, dim=0))
