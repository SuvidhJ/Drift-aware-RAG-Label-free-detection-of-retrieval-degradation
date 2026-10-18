# Human validation: instructions for annotators

Thank you for helping. This takes about 1.5–2 hours. **Please label independently; do not compare
notes until both files are done.**

**Files.**
- Suvidh: `annotator_1.csv`.
- Aryan: `annotator_2.csv`.
- Both files contain the same 200 cases, in a different order.

**What each row shows.**

| Column | What it is |
|---|---|
| `statement` | A fill-in-the-blank question the AI model answered |
| `context` | The passages the model was given. Two of them give conflicting values for the same fact; sometimes they carry dates |
| `model_answer` | What the model replied |
| `value_A`, `value_B` | The two conflicting values from the context, in random order |

**Your task.** Fill in the column `label (A/B/BOTH/NEITHER)` with exactly one of these:

| Label | When to use it |
|---|---|
| **A** | The answer commits to value A only |
| **B** | The answer commits to value B only |
| **BOTH** | The answer gives both values (e.g. "6.17, 6.07" or "6.17 or 6.07") |
| **NEITHER** | The answer gives neither value: "unknown", a different number, or empty |

**Rules.**
- **Units and formatting don't matter.** "1,860,000,000" and "1.86 billion" are the same value.
  So are "6.07 billion" and "6.07", and "−163" and "163 loss".
- **Small rounding counts as a match** if it clearly refers to that value (e.g. 3.951 → "3.95").
- **If the answer is cut off,** label what it actually states.
- **Do not judge which value is correct.** Only judge which value(s) the answer *contains*.
- **Use `comment`** for anything unclear.

Return the completed CSV. Please don't open `KEY_DO_NOT_SHARE.csv`.
