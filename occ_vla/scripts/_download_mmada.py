"""One-off: download MMaDA-8B-MixCoT + the MAGVIT-v2 tokenizer from the
HF Hub. Run inside .venv_mmada."""

from huggingface_hub import snapshot_download

mmada_path = snapshot_download("Gen-Verse/MMaDA-8B-MixCoT")
print("MMADA_DOWNLOADED_TO:", mmada_path)

magvit_path = snapshot_download("showlab/magvitv2")
print("MAGVIT_DOWNLOADED_TO:", magvit_path)
