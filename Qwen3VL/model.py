import os, json, argparse, torch
import numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor

# ── Paths ──────────────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/mahdipou/models/Qwen3-VL-8B-Instruct")
ACTIVATIONS_DIR = "/mnt/mahdipou/models/Anhedonic-qwen3/activations"
NEURONS_JSON    = "neurons_qwen3.json" 

MODEL_LABEL   = "Qwen3VL (Anhedonic Model)" 
DEFAULT_MAX_TOKENS  = 512


class AnhedonicModelA:
    def __init__(self):
        print("=" * 62)
        print(f"  {MODEL_LABEL}")
        print("=" * 62)

        # 1. Neuron indices — from pre-extracted JSON, no CSV dependency
        if not os.path.exists(NEURONS_JSON):
            raise FileNotFoundError(
                f"{NEURONS_JSON} not found. Run your neuron extraction script first."
            )
        with open(NEURONS_JSON) as f:
            # keys are layer indices as strings, values are lists of neuron indices
            self._neuron_map: dict[int, list[int]] = {
                int(k): v for k, v in json.load(f).items()
            }
        self.n_neurons = sum(len(v) for v in self._neuron_map.values())

        # 2. Load model (Qwen3 specific: AutoModelForVision2Seq & trust_remote_code)
        print("  [1/3] Loading model …")
        self._model = AutoModelForVision2Seq.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        self._model.eval()
        
        # Load processor (Qwen3 specific configs)
        self._proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if self._proc.tokenizer.pad_token is None:
            self._proc.tokenizer.pad_token = self._proc.tokenizer.eos_token
        self._proc.tokenizer.padding_side = "left"
        
        self._layers = self._model.model.language_model.layers

        # 3. Neutral means
        print("  [2/3] Computing neutral activation means …")
        mean_acts = self._load_neutral_means()

        # 4. Install permanent hooks
        print("  [3/3] Installing permanent ablation hooks …")
        self._install_hooks(mean_acts)
        print("  ✓ Ready.\n")

    def _load_neutral_means(self) -> np.ndarray:
        parts = []
        for domain in ["geo", "math"]:
            path = os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt")
            data = torch.load(path, map_location="cpu")
            parts.append(torch.stack(list(data.values())).float())
        return torch.cat(parts, dim=0).mean(dim=0).numpy()

    def _install_hooks(self, mean_acts: np.ndarray):
        for layer_idx, neurons in self._neuron_map.items():
            idx   = torch.tensor(neurons).long().to("cuda")
            means = torch.tensor(mean_acts[layer_idx, neurons], dtype=torch.bfloat16).to("cuda")
            def _make(i, m):
                def _hook(module, _in, out):
                    out[:, :, i] = m.unsqueeze(0).unsqueeze(0)
                    return out
                return _hook
            self._layers[layer_idx].mlp.act_fn.register_forward_hook(
                _make(idx, means)
            )



    def generate_response(self, prompt: str,
                          max_new_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        """Send any text prompt; returns the ablated model's response."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        
        text = self._proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Added padding=True based on your Qwen3 snippet
        inputs = self._proc(text=[text], return_tensors="pt", padding=True).to("cuda")
        
        with torch.no_grad():
            gen = self._model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=False
            )
            
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        return self._proc.batch_decode(trimmed, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)[0]

    def chat(self):
        """Interactive chat loop with the ablated model."""
        print(f"{'─'*62}")
        print(f"  {MODEL_LABEL} - Interactive Chat")
        print(f"  Commands: /info  /quit")
        print(f"{'─'*62}\n")
        while True:
            try:
                user_in = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting."); break
            if not user_in: continue
            if user_in.lower() in ("/quit", "/exit"): print("Exiting."); break
            if user_in.lower() == "/info":
                print(f"  Neurons Clamped : {self.n_neurons:,}")
                continue
            
            print(f"\nModel: {self.generate_response(user_in)}\n")


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
    parser = argparse.ArgumentParser(description="Anhedonic Model, Qwen3-VL-8B")
    parser.add_argument("--prompt",     type=str,   default=None)
    parser.add_argument("--max_tokens", type=int,   default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()
    
    if args.prompt:
        print(model.generate_response(args.prompt, max_new_tokens=args.max_tokens))
    else:
        model.chat()