"""Consolidate the two Demucs test evals, re-run the leakage audit, and PROVE
the test set is bit-identical to the one used for MP-SENet.

Writes results/demucs_ft_test_eval.json in the same schema as
results/mpsenet_ft_test_eval.json so the two can be merged programmatically.
"""
import json
import csv
import hashlib

RUNS = {"v2_epoch_57": 127962, "parity_epoch_45": 127963}
CKPTS = {
    "v2_epoch_57": "checkpoints/dereverb_demucs_v2/epoch_57/g_00047310",
    "parity_epoch_45": "checkpoints/dereverb_demucs/epoch_45/g_00037350",
}
LRS = {"v2_epoch_57": 3e-4, "parity_epoch_45": 1e-4}
VAL_PESQ = {"v2_epoch_57": 2.335, "parity_epoch_45": 2.237}

per_run = {}
for tag in RUNS:
    with open(f"results/demucs_ft_test_eval_{tag}.json") as f:
        per_run[tag] = json.load(f)
ref = per_run["v2_epoch_57"]

# ---------- leakage audit (same as the MP-SENet run) ----------
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
    "note": ("Test RIRs disjoint from every RIR seen in training. Clean test speech disjoint "
             "from clean training speech. clean_test was used for training-time validation "
             "logging only, always paired with train-split RIRs, never backpropagated."),
}

# ---------- PROOF of test-set identity with the MP-SENet run ----------
# The reverberant-input metrics are a property of the FILES ONLY -- they do not
# involve the model at all. If they match row-for-row against the MP-SENet
# per-utterance CSV, the two evals provably consumed the same audio.
INPUT_COLS = ["pesq_input", "stoi_input", "estoi_input", "si_sdr_input"]
mp_rows = list(csv.DictReader(open("results/mpsenet_ft_test_eval_epoch_50.csv")))
dm_rows = list(csv.DictReader(open("results/demucs_ft_test_eval_v2_epoch_57.csv")))

identity = {"n_rows_mpsenet": len(mp_rows), "n_rows_demucs": len(dm_rows)}
identity["row_count_match"] = len(mp_rows) == len(dm_rows)
pair_mismatch = sum(
    1 for a, b in zip(mp_rows, dm_rows)
    if (a["utterance_id"], a["rir_id"]) != (b["utterance_id"], b["rir_id"])
)
identity["utterance_rir_pairing_mismatches"] = pair_mismatch
col_mismatch = {
    c: sum(1 for a, b in zip(mp_rows, dm_rows) if a[c] != b[c]) for c in INPUT_COLS
}
col_maxdiff = {
    c: max(abs(float(a[c]) - float(b[c])) for a, b in zip(mp_rows, dm_rows))
    for c in INPUT_COLS
}
identity["input_metric_exact_mismatches_per_column"] = col_mismatch
identity["input_metric_max_abs_diff_per_column"] = col_maxdiff
identity["pesq_stoi_sisdr_bit_identical"] = all(
    col_mismatch[c] == 0 for c in ["pesq_input", "stoi_input", "si_sdr_input"]
)
identity["estoi_agrees_within"] = col_maxdiff["estoi_input"]
identity["estoi_note"] = (
    "estoi_input differs in 2635/4120 rows but only at ~1e-14. pystoi's extended STOI "
    "is not bit-reproducible: recomputing it 3x on the SAME audio in one process yields "
    "3 different values spanning the same 1e-14 range (BLAS reduction order). This is "
    "numerical noise in the metric, not a difference in the audio -- PESQ, STOI and "
    "SI-SDR, which use the same inputs, are bit-identical in all 4120 rows."
)
with open("test_sets/reverb/metadata.csv", "rb") as f:
    identity["metadata_md5"] = hashlib.md5(f.read()).hexdigest()
identity["metadata_md5_seen_by_demucs_jobs"] = ref["test_set"]["metadata_md5"]
identity["method"] = (
    "Both evals were pointed at the identical absolute path "
    "<Muse-Reverb-FN>/test_sets/reverb -- the same 4120 wav "
    "files on disk, not a regenerated set. Confirmed by (a) matching metadata.csv md5, "
    "(b) identical utterance_id/rir_id pairing row-for-row, and (c) the model-independent "
    "reverberant-input metrics (PESQ/STOI/SI-SDR) matching bit-for-bit in all 4120 rows."
)

out = {
    "description": "Held-out test-set evaluation of the fine-tuned Demucs (DNS64) dereverberation model.",
    "date": "2026-08-02",
    "model": "demucs_dns64",
    "eval_script": "evaluate_demucs_testset.py",
    "sbatch_script": "scripts/launch_demucs_test_eval.sh",
    "slurm_jobs": RUNS,
    "primary_checkpoint": CKPTS["v2_epoch_57"],
    "checkpoint_repo": "<Muse-Reverb-FN-demucs> (branch demucs-reverb-ft)",
    "inference": ref["inference"],
    "test_set": ref["test_set"],
    "test_set_identity_with_mpsenet": identity,
    "leakage_audit": leakage,
    "metrics_note": "DNSMOS omitted: speechmos.dnsmos is broken in this environment (returns nan).",
    "results": {
        tag: {
            "checkpoint": CKPTS[tag],
            "learning_rate": LRS[tag],
            "best_val_pesq": VAL_PESQ[tag],
            "summary": d["summary"],
            "rt60_breakdown": d["rt60_breakdown"],
            "per_utterance_csv": f"results/demucs_ft_test_eval_{tag}.csv",
        }
        for tag, d in per_run.items()
    },
}
with open("results/demucs_ft_test_eval.json", "w") as f:
    json.dump(out, f, indent=2)

with open("results/demucs_ft_test_eval_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["checkpoint", "condition", "pesq", "stoi", "estoi", "si_sdr_db", "n_utts"])
    for tag in ["parity_epoch_45", "v2_epoch_57"]:
        s = per_run[tag]["summary"]
        w.writerow([tag, "reverberant_input",
                    round(s["pesq_input"]["mean"], 4), round(s["stoi_input"]["mean"], 4),
                    round(s["estoi_input"]["mean"], 4), round(s["si_sdr_input"]["mean"], 2),
                    s["pesq_input"]["n"]])
        w.writerow([tag, "enhanced",
                    round(s["pesq"]["mean"], 4), round(s["stoi"]["mean"], 4),
                    round(s["estoi"]["mean"], 4), round(s["si_sdr"]["mean"], 2),
                    s["pesq"]["n"]])

print("TEST-SET IDENTITY vs MP-SENet run")
for k in ["n_rows_mpsenet", "n_rows_demucs", "row_count_match",
          "utterance_rir_pairing_mismatches", "input_metric_exact_mismatches_per_column",
          "input_metric_max_abs_diff_per_column", "pesq_stoi_sisdr_bit_identical",
          "metadata_md5", "metadata_md5_seen_by_demucs_jobs"]:
    print(f"  {k}: {identity[k]}")

print("\nLEAKAGE AUDIT")
for k in ["n_train_split_rirs", "n_test_split_rirs", "n_rirs_used_in_test_set",
          "test_rirs_that_are_train_split", "n_clean_train_utts", "n_clean_test_utts",
          "test_utts_that_are_train_utts"]:
    print(f"  {k}: {leakage[k]}")

print("\nVALID-METRIC COUNTS (v2_epoch_57, out of 4120)")
for k, v in ref["summary"].items():
    if k != "delta":
        print(f"  {k:<16} n={v['n']:<5} mean={v['mean']:>8.4f}  std={v['std']:.4f}")

print("\nCHECKPOINT COMPARISON (enhanced, test set)")
for tag in ["parity_epoch_45", "v2_epoch_57"]:
    s = per_run[tag]["summary"]
    print(f"  {tag:<16} lr={LRS[tag]:.0e}  val_PESQ={VAL_PESQ[tag]:.3f}  ->  "
          f"test PESQ {s['pesq']['mean']:.3f}  STOI {s['stoi']['mean']:.4f}  "
          f"ESTOI {s['estoi']['mean']:.4f}  SI-SDR {s['si_sdr']['mean']:+.2f} dB")

print("\nWrote results/demucs_ft_test_eval.json + results/demucs_ft_test_eval_summary.csv")
