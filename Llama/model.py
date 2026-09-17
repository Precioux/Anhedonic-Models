import os, json, argparse, torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH      = os.environ.get("MODEL_PATH", "/mnt/mahdipou/models/Llama-3.1-8B-Instruct")
ACTIVATIONS_DIR = "/mnt/mahdipou/models/LLM/activations_llama"
NEURONS_JSON    = "neurons_llama.json"

MODEL_LABEL     = "Llama-3.1-8B-Instruct"
TOTAL_NEURONS   = 32 * 14336  # Standard approx intermediate size for Llama 3 8B

DEFAULT_MAX_TOKENS  = 250


class AnhedonicModelA:
    def __init__(self):
        print("=" * 62)
        print(f"  {MODEL_LABEL}")
        print("=" * 62)

        # 1. Neuron indices 
        if not os.path.exists(NEURONS_JSON):
            raise FileNotFoundError(
                f"{NEURONS_JSON} not found. Ensure the path is correct."
            )
        with open(NEURONS_JSON) as f:
            self._neuron_map: dict[int, list[int]] = {
                int(k): v for k, v in json.load(f).items()
            }
        self.n_neurons = sum(len(v) for v in self._neuron_map.values())

        # 2. Load model
        print("  [1/3] Loading model …")
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True
        )
        self._model.eval()
        self._layers = self._model.model.layers

        # 3. Neutral means
        print("  [2/3] Computing neutral activation means …")
        mean_acts = self._load_neutral_means()

        # 4. Install permanent hooks
        print("  [3/3] Installing permanent ablation hooks …")
        self._install_hooks(mean_acts)
        pct = (self.n_neurons / TOTAL_NEURONS) * 100
        print(f"        {self.n_neurons:,} neurons clamped (approx. {pct:.4f}% of network)")
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
            if not neurons:
                continue
            idx   = torch.tensor(neurons).long().to("cuda")
            means = torch.tensor(mean_acts[layer_idx, neurons], dtype=torch.bfloat16).to("cuda")
            
            def _make(i, m):
                def _hook(module, _in, out):
                    out[:, :, i] = m.unsqueeze(0).unsqueeze(0)
                    return out
                return _hook
                
            self._layers[layer_idx].mlp.act_fn.register_forward_hook(_make(idx, means))

    def generate_response(self, prompt: str,
                          max_new_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        """Send any text prompt; returns the ablated model's response."""
        messages = [
            {
                "role": "system", 
                "content": "You are a participant in this experiment. You strictly follow rules, keep your reasoning very brief (1 or 2 sentences max), and always output in the exact requested format."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            gen = self._model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,       
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=self._tokenizer.eos_token_id
            )
            
        return self._tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def chat(self):
        """Interactive chat loop with the ablated model."""
        print(f"{'─'*62}")
        print(f"  {MODEL_LABEL} (Interactive Chat)")
        print(f"{'─'*62}\n")
        while True:
            try:
                user_in = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting."); break
                
            if not user_in: 
                continue
            if user_in.lower() in ("/quit", "/exit"): 
                print("Exiting."); break
            if user_in.lower() == "/info":
                print(f"  Neurons   : {self.n_neurons:,}  (approx. {self.n_neurons/TOTAL_NEURONS*100:.4f}%)")
                continue
                
            print(f"\nModel A: {self.generate_response(user_in)}\n")


# ── Module-level API ───────────────────────────────────────────────────────
model: "AnhedonicModelA" = None  

def _init():
    global model
    if model is None:
        model = AnhedonicModelA()

def generate(prompt: str, **kwargs) -> str:
    _init()
    return model.generate_response(prompt, **kwargs)


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anhedonic Model, Llama-3.1-8B-Instruct")
    parser.add_argument("--prompt",     type=str,   default=None)
    parser.add_argument("--max_tokens", type=int,   default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()
    
    _init()
    
    if args.prompt:
        print(model.generate_response(args.prompt, max_new_tokens=args.max_tokens))
    else:
        model.chat()