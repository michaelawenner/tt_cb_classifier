import os
import pandas as pd
from src.load_data import load_definitions, load_labeled_examples
from src.prompts import build_prompts
from src.classifier import classify_project

# --- CONFIG ---
PROVIDER    = "apertus"                          # <-- add this; classifier needs it
MODEL_ID    = "swiss-ai/apertus-v1.5-70b"        # pin the dated ID for the cycle
MODEL_DELAY = 0.7                                 # 100/min = 0.6s floor; 0.7 gives headroom
NUM_EXAMPLES = 14                                 # <-- was 9; you now have 14 examples
INPUT_PATH  = "data/20260814_TRIPS.csv"   # <-- was llm_input.csv; you renamed it
OUTPUT_PATH = "results/TRIPS_classified_projects.csv"
CHECKPOINT  = "results/TRIPS_classified_checkpoint.csv"

# --- Load inputs ---
print("📥 Loading data...")
definitions = load_definitions()
train_df    = load_labeled_examples()

df = pd.read_csv(INPUT_PATH)

# --- Build prompts (same examples/definitions as validation) ---
context_tt, role_tt, context_cb, role_cb = build_prompts(train_df, definitions, num_examples=NUM_EXAMPLES)

# --- Resume from checkpoint if present ---
if os.path.exists(CHECKPOINT):
    done = pd.read_csv(CHECKPOINT)
    # only count rows with BOTH predictions resolved as truly done
    resolved = done[done["is_TT"].notna() & done["is_CB"].notna()]
    unresolved_keys = set(done["label"]) - set(resolved["label"])
    print(f"⏩ Resuming: {len(resolved)} done, {len(unresolved_keys)} to retry")
    done_keys = set(resolved["label"])
    todo = df[~df["label"].isin(done_keys)].copy()
    results = [resolved]          # <-- keep only resolved rows; failed ones will be redone
else:
    todo = df.copy()
    results = []

print(f"🚀 Classifying {len(todo)} projects with {MODEL_ID}...")

# --- Classify, checkpointing every N rows ---
CHECKPOINT_EVERY = 25
buffer = []
for i, (_, row) in enumerate(todo.iterrows(), 1):
    tt = classify_project(row["Title"], row["Description"], context_tt, role_tt, "TT",
                      provider=PROVIDER, model=MODEL_ID, delay=MODEL_DELAY)
    cb = classify_project(row["Title"], row["Description"], context_cb, role_cb, "CB",
                      provider=PROVIDER, model=MODEL_ID, delay=MODEL_DELAY)
    buffer.append({**row.to_dict(), "is_TT": tt, "is_CB": cb})

    if i % CHECKPOINT_EVERY == 0 or i == len(todo):
        chunk = pd.DataFrame(buffer)
        combined = pd.concat(results + [chunk], ignore_index=True) if results else chunk
        combined.to_csv(CHECKPOINT, index=False)
        print(f"  💾 checkpoint: {len(combined)} done")

# --- Finalize ---
final = pd.read_csv(CHECKPOINT)
final.to_csv(OUTPUT_PATH, index=False)
n_fail = final["is_TT"].isna().sum() + final["is_CB"].isna().sum()
print(f"✅ Done. {len(final)} projects → {OUTPUT_PATH}")
if n_fail:
    print(f"⚠️ {n_fail} unresolved (None) predictions — re-run to retry those rows")