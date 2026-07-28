"""Stage 11: faithfulness artifacts — logit-lens on v_conv and the phase/steering
directions (+ SAE decomposition when running the Llama-8B fork)."""
from __future__ import annotations

import json
import time

from ..generation.hf_backend import load_model_and_tokenizer
from ..interp.logit_lens import direction_vocab
from ..probes.probe import LinearProbe
from ..steering.vectors import load_vector
from ._stage import mark_done, setup, stage_args


def main():
    args = stage_args(__doc__)
    cfg, paths, stage_dir = setup(args, "interp")
    t0 = time.time()
    model, tok = load_model_and_tokenizer(cfg.model.hf_id, cfg.model.dtype,
                                          cfg.model.device)
    out = {}
    conv_path = paths.probes("conv", cfg.probe.arch) / f"L{cfg.model.probe_layer}.pt"
    if conv_path.exists():
        probe = LinearProbe.load(conv_path)
        out["v_conv"] = direction_vocab(model, tok, probe.direction)
    for vec_file in sorted(paths.steering().glob("*_L*.pt")):
        blob = load_vector(vec_file)
        out[vec_file.stem] = direction_vocab(model, tok, blob["v"])
    if "llama" in cfg.model.tag:
        try:
            from ..interp.sae import decompose_direction, load_sae_decoder
            W = load_sae_decoder()
            if conv_path.exists():
                out["v_conv_sae_features"] = decompose_direction(
                    LinearProbe.load(conv_path).direction, W)
        except Exception as e:
            out["sae_error"] = str(e)
    out_path = paths.analysis() / "interp.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    for name, d in out.items():
        if isinstance(d, dict) and "promoted" in d:
            print(f"{name} promotes: {[t for t, _ in d['promoted'][:12]]}")
    mark_done(stage_dir, t0)


if __name__ == "__main__":
    main()
