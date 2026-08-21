import argparse


# Same hardware-sizing ceiling src/ilm_profile.py's resolve_resource_config()
# uses for n_embd on consumer hardware (`min(768, ...)`) -- kept in sync here
# rather than duplicated as an unrelated-looking literal. MAX_LAYERS has no
# equivalent upstream constant; 12 is this heuristic's own conservative cap,
# chosen so growth stays local-PC friendly rather than requesting a network
# size that would no longer train in reasonable time on a single machine.
MAX_LAYERS = 12
MAX_EMBD = 768


def suggest_growth(current_val_loss: float, target_val_loss: float, n_layer: int, n_embd: int) -> tuple[int, int]:
    """Simple growth heuristic to request more 'neurons' when quality is below target."""
    if current_val_loss <= target_val_loss:
        return n_layer, n_embd

    # grow conservatively to stay local-PC friendly
    new_layers = min(n_layer + 1, MAX_LAYERS)
    new_embd = min(int(n_embd * 1.25), MAX_EMBD)
    return new_layers, new_embd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-loss", type=float, required=True)
    parser.add_argument("--target-loss", type=float, default=1.8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--embd", type=int, default=192)
    args = parser.parse_args()

    nl, ne = suggest_growth(args.val_loss, args.target_loss, args.layers, args.embd)
    if (nl, ne) == (args.layers, args.embd):
        print("No growth suggested. Current network size is sufficient.")
    else:
        print(f"Growth suggested -> layers: {args.layers} -> {nl}, embd: {args.embd} -> {ne}")


if __name__ == "__main__":
    main()
