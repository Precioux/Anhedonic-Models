
import os, json, argparse, torch
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2-VL-7B-Instruct")
ACTIVATIONS_DIR = "/mnt/mahdipou/anhedonia/extraction/outputs/activations"
NEURONS_JSON    = "neurons_qwen2vl.json"  

MODEL_LABEL   = "Qwen2VL"
TOTAL_NEURONS = 28 * 18944

DEFAULT_MAX_TOKENS  = 512


class AnhedonicModelA:
    def __init__(self):
        print("=" * 62)
        print(f"  {MODEL_LABEL}")
        print("=" * 62)

        # 1. Neuron indices — from pre-extracted JSON, no CSV dependency
        if not os.path.exists(NEURONS_JSON):
            raise FileNotFoundError(
                f"{NEURONS_JSON} not found. Run `python extract_neurons.py` first."
            )
        with open(NEURONS_JSON) as f:
            # keys are layer indices as strings, values are lists of neuron indices
            self._neuron_map: dict[int, list[int]] = {
                int(k): v for k, v in json.load(f).items()
            }
        self.n_neurons = sum(len(v) for v in self._neuron_map.values())

        # 2. Load model
        print("  [1/3] Loading model …")
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self._model.eval()
        self._proc   = AutoProcessor.from_pretrained(MODEL_PATH)
        self._layers = self._model.model.language_model.layers

        # 3. Neutral means
        print("  [2/3] Computing neutral activation means …")
        mean_acts = self._load_neutral_means()

        # 4. Install permanent hooks
        print("  [3/3] Installing permanent ablation hooks …")
        self._install_hooks(mean_acts)
        pct = self.n_neurons / TOTAL_NEURONS * 100
        print(f"        {self.n_neurons:,} neurons clamped ({pct:.4f}% of network)")
        print("  ✓ Ready.\n")

    def _load_neutral_means(self) -> np.ndarray:
        parts = []
        for domain in ["geo", "math"]:
            path = os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt")
            data = torch.load(path, map_location="cpu")
            parts.append(torch.stack(list(data.values())).float())
        return torch.cat(parts, dim=0).mean(dim=0).numpy()   # [28, 18944]

    def _install_hooks(self, mean_acts: np.ndarray):
        for layer_idx, neurons in self._neuron_map.items():
            idx   = torch.tensor(neurons).long().to("cuda")
            means = torch.tensor(mean_acts[layer_idx, neurons], dtype=torch.bfloat16).to("cuda")
            def _make(i, m):
                def _hook(module, _in, out):
                    out[:, -1, i] = m.unsqueeze(0).unsqueeze(0)
                    return out
                return _hook
            self._layers[layer_idx].mlp.act_fn.register_forward_hook(
                _make(idx, means)
            )

    def generate_response(self, prompt: str,
                          max_new_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        """Send any text prompt; returns the ablated model's response."""
        text = self._proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            tokenize=False, add_generation_prompt=True
        )
        inputs = self._proc(text=[text], return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        return self._proc.batch_decode(trimmed, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)[0]

    def chat(self):
        """Interactive chat loop with the ablated model."""
        print(f"{'─'*62}")
        print(f"  {MODEL_LABEL}")
        print(f"{'─'*62}\n")
        while True:
            try:
                user_in = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting."); break
            if not user_in: continue
            if user_in.lower() in ("/quit", "/exit"): print("Exiting."); break
            if user_in.lower() == "/info":
                print(f"  Ablation  : layers 18–27")
                print(f"  Neurons   : {self.n_neurons:,}  ({self.n_neurons/TOTAL_NEURONS*100:.4f}%)")
                continue
            print(f"\nModel A: {self.generate_response(user_in)}\n")


# ── Module-level API ───────────────────────────────────────────────────────
model: "AnhedonicModelA" = None  # type: ignore

def _init():
    global model
    if model is None:
        model = AnhedonicModelA()

_init()

def generate(prompt: str, **kwargs) -> str:
    return model.generate_response(prompt, **kwargs)


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anhedonic Model , Qwen2VL-7b")
    parser.add_argument("--prompt",     type=str,   default=None)
    parser.add_argument("--max_tokens", type=int,   default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()
    if args.prompt:
        print(model.generate_response(args.prompt, max_new_tokens=args.max_tokens))
    else:
        model.chat()