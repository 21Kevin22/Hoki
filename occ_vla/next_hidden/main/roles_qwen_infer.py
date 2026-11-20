"""C7 step 2 (runs in third_party/.venv_qwen, ONE GPU, never at the same time as pi0.5 -- PREREG_main.md §5, §8).
Qwen2.5-VL-3B-Instruct on every t0 image (main/qwen/t0/*.png) with the pre-registered prompt (PREREG change log,
2026-11-20, C7), greedy decoding, fp16, sdpa attention. Writes main/qwen/qwen_roles.json (raw text + parsed JSON +
model snapshot) and frees the GPU at exit.
Usage: CUDA_VISIBLE_DEVICES=0 third_party/.venv_qwen/bin/python main/roles_qwen_infer.py"""
import glob, json, os, re, sys, time
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks import TASKS   # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
PROMPT = ("You see a robot workspace. The robot's instruction is: '{instruction}'. Find the container or piece of furniture "
          "that the instruction acts on (for example a microwave, one drawer of a cabinet, a compartment of a caddy). "
          "Answer ONLY with JSON: {{\"container_box\": [x1, y1, x2, y2], \"handle_point\": [x, y] or null, "
          "\"front_box\": [x1, y1, x2, y2]}}. container_box: the whole container in pixels. handle_point: a point on the handle "
          "used to open or close it, null if there is none. front_box: the door, drawer front or opening where objects go in.")


QWEN_SIDE = 280      # the processor resizes the 256x256 image to 280x280 (multiple of 28); answers use that space
SCALE = 256.0 / QWEN_SIDE


def parse(text):
    """JSON -> boxes/point rescaled from Qwen's 280x280 input space to our 256x256 pixels, then range-checked."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    num = lambda b, n: isinstance(b, list) and len(b) == n and all(isinstance(v, (int, float)) for v in b)
    for k, n in (("container_box", 4), ("front_box", 4), ("handle_point", 2)):
        if num(j.get(k), n):
            j[k] = [v * SCALE for v in j[k]]
    ok = lambda b, n: num(b, n) and all(0 <= v <= 256 for v in b)
    if not ok(j.get("container_box"), 4) or not ok(j.get("front_box"), 4):
        return None
    if j.get("handle_point") is not None and not ok(j.get("handle_point"), 2):
        j["handle_point"] = None
    return j


def main():
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float16,
                                                               attn_implementation="sdpa").to("cuda").eval()
    proc = AutoProcessor.from_pretrained(path, min_pixels=256 * 256, max_pixels=256 * 256)
    out, t0 = {}, time.time()
    for png in sorted(glob.glob(os.path.join(HERE, "main", "qwen", "t0", "*.png"))):
        stem = os.path.basename(png)[:-4]
        task = stem.split("_")[0]
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": PROMPT.format(instruction=TASKS[task][2])}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[Image.open(png).convert("RGB")], return_tensors="pt").to("cuda")
        with torch.no_grad():
            ids = model.generate(**inputs, do_sample=False, max_new_tokens=200)
        raw = proc.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        out[stem] = dict(raw=raw, parsed=parse(raw))
        print(stem, "ok" if out[stem]["parsed"] else "UNPARSED", flush=True)
    json.dump(dict(model=MODEL, snapshot=os.path.basename(path), prompt=PROMPT, decoding="greedy, max_new_tokens=200",
                   dtype="float16", attn="sdpa", seconds=round(time.time() - t0, 1),
                   peak_mem_mib=round(torch.cuda.max_memory_allocated() / 2**20), results=out),
              open(os.path.join(HERE, "main", "qwen", "qwen_roles.json"), "w"), indent=1)
    n_bad = sum(v["parsed"] is None for v in out.values())
    print(f"done: {len(out)} images, unparsed {n_bad}", flush=True)


def reparse():
    """Re-parse the saved raw answers (no new inference)."""
    path = os.path.join(HERE, "main", "qwen", "qwen_roles.json")
    J = json.load(open(path))
    for v in J["results"].values():
        v["parsed"] = parse(v["raw"])
    J["coord_space"] = f"Qwen input {QWEN_SIDE}x{QWEN_SIDE}, rescaled to 256x256"
    json.dump(J, open(path, "w"), indent=1)
    print("reparsed:", len(J["results"]), "unparsed", sum(v["parsed"] is None for v in J["results"].values()))


if __name__ == "__main__":
    reparse() if "--reparse" in sys.argv else main()
