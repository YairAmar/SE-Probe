"""Consolidate per-epoch test-eval JSONs into one summary + run a leakage audit."""
import json
import csv

EPOCHS = {"epoch_50": 127912, "epoch_49": 127913, "epoch_48": 127914}

per_epoch = {}
for e in EPOCHS:
    with open(f"results/mpsenet_ft_test_eval_{e}.json") as f:
        per_epoch[e] = json.load(f)

ref = per_epoch["epoch_50"]

# --- leakage audit ---
split = json.load(open("data/rir_split.json"))
train_rirs = {k for k, v in split.items() if v == "train"}
test_rirs = {k for k, v in split.items() if v == "test"}

rows = list(csv.DictReader(open("test_sets/reverb/metadata.csv")))
used_rirs = {r["rir_id"] for r in rows}
used_utts = {r["utterance_id"] for r in rows}

train_utts = {l.split("|")[0] for l in open("VoiceBank+DEMAND/training.txt").read().split() if l}
test_utts = {l.split("|")[0] for l in open("VoiceBank+DEMAND/test.txt").read().split() if l}

leakage = {
    "n_train_split_rirs": len(train_rirs),
    "n_test_split_rirs": len(test_rirs),
    "n_rirs_used_in_test_set": len(used_rirs),
    "test_rirs_that_are_train_split": len(used_rirs & train_rirs),
    "n_clean_train_utts": len(train_utts),
    "n_clean_test_utts": len(test_utts),
    "test_utts_that_are_train_utts": len(used_utts & train_utts),
    "training_data": "data/VB_DEMAND_16K/clean_train x train-split RIRs (95% of 798; other 5% held out for val)",
    "training_val_data": "data/VB_DEMAND_16K/clean_test x val-holdout RIRs (drawn from the TRAIN split)",
    "note": (
        "Test RIRs are fully disjoint from every RIR seen during training (including the "
        "validation holdout, which came from the train split). Clean test speech is disjoint "
        "from clean training speech. The clean_test utterances were used for training-time "
        "validation logging, but always paired with train-split RIRs and never backpropagated; "
        "epoch_50 is the final epoch, not a checkpoint selected on validation."
    ),
}

out = {
    "description": "Held-out test-set evaluation of the fine-tuned MP-SENet dereverberation model.",
    "date": "2026-08-02",
    "eval_script": "evaluate_mpsenet_testset.py",
    "sbatch_script": "scripts/launch_test_eval.sh",
    "slurm_jobs": EPOCHS,
    "primary_checkpoint": "checkpoints_mpsenet/all_v2/epoch_50/g_00295909",
    "config_path": "checkpoints_mpsenet/all_v2/config.json",
    "config": ref["config_stft"],
    "test_set": ref["test_set"],
    "leakage_audit": leakage,
    "metrics_note": "DNSMOS omitted: speechmos.dnsmos is broken in this environment (returns nan for any input).",
    "results": {
        e: {
            "summary": d["summary"],
            "rt60_breakdown": d["rt60_breakdown"],
            "per_utterance_csv": f"results/mpsenet_ft_test_eval_{e}.csv",
        }
        for e, d in per_epoch.items()
    },
}

with open("results/mpsenet_ft_test_eval.json", "w") as f:
    json.dump(out, f, indent=2)

with open("results/mpsenet_ft_test_eval_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "condition", "pesq", "stoi", "estoi", "si_sdr_db", "n_utts"])
    for e in ["epoch_48", "epoch_49", "epoch_50"]:
        s = per_epoch[e]["summary"]
        w.writerow([e, "reverberant_input",
                    round(s["pesq_input"]["mean"], 4), round(s["stoi_input"]["mean"], 4),
                    round(s["estoi_input"]["mean"], 4), round(s["si_sdr_input"]["mean"], 2),
                    s["pesq_input"]["n"]])
        w.writerow([e, "enhanced",
                    round(s["pesq"]["mean"], 4), round(s["stoi"]["mean"], 4),
                    round(s["estoi"]["mean"], 4), round(s["si_sdr"]["mean"], 2),
                    s["pesq"]["n"]])

print("LEAKAGE AUDIT")
for k in ["n_train_split_rirs", "n_test_split_rirs", "n_rirs_used_in_test_set",
          "test_rirs_that_are_train_split", "n_clean_train_utts", "n_clean_test_utts",
          "test_utts_that_are_train_utts"]:
    print(f"  {k}: {leakage[k]}")

print("\nVALID-METRIC COUNTS / SPREAD (epoch_50, out of 4120):")
for k, v in ref["summary"].items():
    if k != "delta":
        print(f"  {k:<16} n={v['n']:<5} mean={v['mean']:>8.4f}  std={v['std']:.4f}")

print("\nEPOCH COMPARISON (enhanced):")
for e in ["epoch_48", "epoch_49", "epoch_50"]:
    s = per_epoch[e]["summary"]
    print(f"  {e}: PESQ {s['pesq']['mean']:.3f}  STOI {s['stoi']['mean']:.4f}  "
          f"ESTOI {s['estoi']['mean']:.4f}  SI-SDR {s['si_sdr']['mean']:+.2f} dB")

print("\nWrote results/mpsenet_ft_test_eval.json + results/mpsenet_ft_test_eval_summary.csv")
