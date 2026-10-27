"""
Shared CLI for the systematic-aware (systematics-adversarial) VBF DNN loss.

train_dnn.py, hpo_optuna.py and hpo_ray.py all take these flags; hpo_ray.py forwards them to
its GPU workers via adversarial_argv(). Pure argparse (no torch), so hpo_ray.py submit can use
it from the AF global env. Method and how to run: docs/vbf_dnn_systematic_aware.md.
"""
import argparse
from typing import List


def add_adversarial_args(p: argparse.ArgumentParser) -> None:
    adv = p.add_argument_group("systematics-adversarial loss")
    adv.add_argument(
        "--use_adversarial",
        action="store_true",
        help=(
            "Add the systematics-consistency term to the loss. Requires --data-dir to "
            "point at fold parquets produced with "
            "preprocess_dnn.py --include-systematic-variations. When this flag is "
            "absent the training path is unchanged."
        ),
    )
    adv.add_argument(
        "--adversarial-lambda",
        type=float,
        default=0.0,
        help="Weight of the adversarial sum (default 0.0).",
    )
    adv.add_argument(
        "--adversarial-detach-nominal",
        action="store_true",
        help=(
            "Stop-gradient on the nominal prediction used as the consistency target, "
            "so only the variation branch is penalised. Default: not detached "
            "(both branches are penalised)."
        ),
    )
    adv.add_argument(
        "--adversarial-consistency-smoothing",
        choices=["inherit", "none"],
        default="inherit",
        help=(
            "Label smoothing inside original_loss(pred_i, pred_nominal). 'inherit' "
            "(default) reuses cfg.label_smoothing, i.e. original_loss literally; "
            "'none' disables it, since the target is already a soft probability and "
            "smoothing it biases variation scores toward 0.5."
        ),
    )
    adv.add_argument(
        "--adversarial-label-only",
        action="store_true",
        help=(
            "STEP 2 of the three-step schedule: use "
            "lambda * sum_i original_loss(pred_i, label) only -- the variations are "
            "trained against the truth label and the consistency term is absent."
        ),
    )
    adv.add_argument(
        "--adversarial-mask-label-term",
        action="store_true",
        help=(
            "Apply --adversarial-high-score-cut to the original_loss(pred_i, label) "
            "term as well, not just to the consistency term. The three-step schedule "
            "puts [score_cut] on every term inside the lambda sum; the earlier "
            "variant-B formula masked the consistency term alone. No effect without "
            "a cut."
        ),
    )
    adv.add_argument(
        "--adversarial-high-score-cut",
        type=float,
        default=None,
        help=(
            "VARIANT B: restrict the consistency term to events with "
            "arctanh(p_nominal) > CUT (e.g. 2.0, i.e. p_nominal > 0.9640). The "
            "significance comes from the top score bins, so decorrelating across the "
            "bulk spends discrimination where it cannot pay. Changes the effective "
            "lambda scale, so re-scan lambda when enabling it."
        ),
    )
    adv.add_argument(
        "--adversarial-variation-chunk",
        type=int,
        default=0,
        help=(
            "Variations per forward/backward chunk (gradient accumulation). "
            "<=0 (default) processes all variations in one chunk; lower it if the "
            "full-variation run runs out of device memory."
        ),
    )


def adversarial_argv(args: argparse.Namespace) -> List[str]:
    """The adversarial flags of `args` as argv, for forwarding to a subprocess. [] when off."""
    if not args.use_adversarial:
        return []
    out = [
        "--use_adversarial",
        "--adversarial-lambda", repr(float(args.adversarial_lambda)),
        "--adversarial-consistency-smoothing", args.adversarial_consistency_smoothing,
        "--adversarial-variation-chunk", str(int(args.adversarial_variation_chunk)),
    ]
    if args.adversarial_detach_nominal:
        out.append("--adversarial-detach-nominal")
    if args.adversarial_label_only:
        out.append("--adversarial-label-only")
    if args.adversarial_mask_label_term:
        out.append("--adversarial-mask-label-term")
    if args.adversarial_high_score_cut is not None:
        out += ["--adversarial-high-score-cut", repr(float(args.adversarial_high_score_cut))]
    return out
