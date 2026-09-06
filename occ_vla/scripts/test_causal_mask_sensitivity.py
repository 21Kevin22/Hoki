"""occ_vla addition (2026-09-06), per user request: counterfactual causal-
sensitivity test for ObjectCentricZeroInitAdapter, following the "causal
confusion" diagnostic pattern (de Haan, Jayaraman, Levine, NeurIPS 2019,
arXiv:1905.11979) -- does NOT retrain anything. Loads the REAL (unablated,
real-mask-trained) task2 adapter checkpoint and, on ONE fixed real held-out
frame, feeds it four different object_mask variants:

  real    -- the true saved per-patch object-of-interest mask (as trained on)
  shifted -- the SAME mask rolled to a different spatial region of the
             256-token grid (same information content/coverage fraction,
             wrong location)
  zero    -- constant all-zero (matches --ablation-mask-mode zero)
  random  -- independent U(0,1) noise, same shape (matches --ablation-mask-mode random)

If the adapter is causally using the mask's actual spatial content, the
predicted action should change measurably more between real-vs-shifted than
between two independent real-vs-real calls (the established GPU/attention-
kernel non-determinism noise floor, see CLAUDE.md's "Real model wiring
implemented" entry). If real-vs-shifted sits inside the real-vs-real noise
floor, that's direct evidence the adapter's output does not track WHERE the
mask says the object is -- consistent with (not just suggestive of, like the
zero-ablation retraining result) the adapter having learned a
position-independent constant bias.
"""
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
OFT_ROOT = os.path.normpath(os.path.join(SCRIPTS_DIR, "..", "thirdparty", "openvla-oft"))
sys.path.insert(0, OFT_ROOT)
os.chdir(OFT_ROOT)

import json  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

from experiments.robot.libero.run_libero_eval import GenerateConfig  # noqa: E402
from experiments.robot.openvla_utils import get_vla, get_processor, get_action_head, get_proprio_projector  # noqa: E402
from prismatic.extern.hf.modeling_prismatic import ObjectCentricZeroInitAdapter  # noqa: E402
from train_object_centric_adapter import load_sample  # noqa: E402

CHECKPOINT = "/home/ubuntu/slocal/Hoki/occ_vla/checkpoints/openvla-7b-oft-libero10-vjepa"
ADAPTER_DIR = "/home/ubuntu/slocal/Hoki/occ_vla/scripts/object_centric_adapter_task2_lora01/step600"
LORA_DIR = "/home/ubuntu/slocal/Hoki/occ_vla/scripts/object_centric_adapter_task2_lora01/step600"
DATA_DIR = "/home/ubuntu/slocal/Hoki/occ_vla/thirdparty/openvla-oft/month2_pairs_task2_n15"
ENTRY_UID = "task2_ep14_t00018"  # corrected=True held-out frame


def main():
    manifest = json.load(open(os.path.join(DATA_DIR, "manifest.json")))
    entry = next(e for e in manifest if e["uid"] == ENTRY_UID)
    print(f"using entry {ENTRY_UID}, object_mask_coverage_frac={entry['object_mask_coverage_frac']:.3f}")

    cfg = GenerateConfig(
        pretrained_checkpoint=CHECKPOINT,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True,
        load_in_8bit=False, load_in_4bit=False,
        center_crop=True, num_open_loop_steps=8, task_suite_name="libero_10", seed=7,
    )
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    device = vla.device
    unnorm_key = next(iter(vla.norm_stats.keys())) if len(vla.norm_stats) == 1 else "libero_10_no_noops"
    cfg.unnorm_key = unnorm_key

    adapter = ObjectCentricZeroInitAdapter(llm_dim=vla.llm_dim).to(vla.device, dtype=torch.bfloat16)
    sd = torch.load(os.path.join(ADAPTER_DIR, "object_centric_adapter_weights.pt"), map_location=vla.device)
    missing, unexpected = adapter.load_state_dict(sd, strict=True)
    print(f"loaded adapter: missing={len(missing)} unexpected={len(unexpected)}")
    vla.object_centric_adapter = adapter

    # occ_vla addition (2026-09-06, priority (3) test): optionally also load
    # the narrow LoRA saved alongside this checkpoint by
    # train_object_centric_adapter_with_lora.py -- LORA_DIR=None (default)
    # reproduces the exact prior (adapter-only) test byte-for-byte.
    if LORA_DIR is not None:
        from peft import LoraConfig, get_peft_model
        lora_sd = torch.load(os.path.join(LORA_DIR, "lora_weights.pt"), map_location=vla.device)
        lora_a_shapes = [v.shape for k, v in lora_sd.items() if "lora_A" in k]
        inferred_rank = lora_a_shapes[0][0]
        lora_layer_idxs = sorted({int(k.split(".layers.")[1].split(".")[0]) for k in lora_sd if ".layers." in k})
        lora_config = LoraConfig(
            r=inferred_rank, lora_alpha=inferred_rank * 2, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            layers_to_transform=lora_layer_idxs, layers_pattern="layers", task_type=None,
        )
        vla.language_model = get_peft_model(vla.language_model, lora_config)
        missing_lm, unexpected_lm = vla.language_model.load_state_dict(lora_sd, strict=False)
        assert len(unexpected_lm) == 0, f"unexpected LoRA keys: {unexpected_lm[:5]}"
        print(f"loaded lora: layers={lora_layer_idxs} rank={inferred_rank} tensors={len(lora_sd)}")

    device = vla.device
    input_ids, attention_mask, pixel_values, real_mask = load_sample(entry, DATA_DIR, processor, cfg, device)
    agent_mask_256 = real_mask[0, :256, 0].float().cpu().numpy()

    def build_mask(agent_part):
        wrist_part = np.zeros_like(agent_part)
        m = np.concatenate([agent_part, wrist_part])[:, None]
        return torch.tensor(m, device=device, dtype=torch.bfloat16).unsqueeze(0)

    variants = {
        "real": build_mask(agent_mask_256),
        "shifted": build_mask(np.roll(agent_mask_256, shift=128)),  # same content, opposite side of the 16x16 grid
        "zero": build_mask(np.zeros_like(agent_mask_256)),
        "random": build_mask(np.random.RandomState(0).uniform(0.0, 1.0, size=agent_mask_256.shape).astype(np.float32)),
    }
    proprio = np.array(entry["state"], dtype=np.float32)

    def get_action(mask):
        with torch.no_grad():
            actions, _ = vla.predict_action(
                input_ids=input_ids, unnorm_key=cfg.unnorm_key, proprio=proprio,
                proprio_projector=proprio_projector, action_head=action_head,
                use_film=cfg.use_film, pixel_values=pixel_values, attention_mask=attention_mask,
                object_mask=mask,
            )
        return np.asarray(actions)[0]  # first chunk step, (7,)

    print("\n--- noise floor: two independent calls with the IDENTICAL real mask ---")
    a_real_1 = get_action(variants["real"])
    a_real_2 = get_action(variants["real"])
    noise_floor = np.linalg.norm(a_real_1 - a_real_2)
    print(f"real call 1: {a_real_1}")
    print(f"real call 2: {a_real_2}")
    print(f"||real1 - real2|| (noise floor) = {noise_floor:.5f}")

    print("\n--- counterfactual: same frame, different mask content ---")
    results = {"real": a_real_1}
    for name in ["shifted", "zero", "random"]:
        a = get_action(variants[name])
        d = np.linalg.norm(a - a_real_1)
        results[name] = a
        ratio = d / noise_floor if noise_floor > 1e-9 else float("inf")
        print(f"{name:8s}: action={a}  ||{name}-real||={d:.5f}  ({ratio:.2f}x noise floor)")

    print("\n=== VERDICT ===")
    print("If shifted/zero/random distances are ~1x the noise floor, the adapter's")
    print("output does NOT causally track the mask's spatial content (consistent")
    print("with the retrained-zero-ablation finding). If shifted is clearly larger")
    print("than the noise floor (and larger than zero/random), the adapter DOES use")
    print("real spatial content from the mask.")


if __name__ == "__main__":
    main()
