# contexto-ish

A small browser word game in the style of [Contexto](https://contexto.me): guess the secret word, and every guess is scored **0–100** by how close its *meaning* is to the answer. Pure HTML/CSS/JS — no build step, no server, no API keys.

The answer is hard-coded to `"answer"` in `game.js`.

## Run it

A local server is required (browsers block ES modules over `file://`):

```bash
python -m http.server 5173
# then open http://127.0.0.1:5173/
```

Any static server works — `npx serve`, `npx http-server`, VS Code Live Server, etc.

The first load downloads a ~25 MB language model; after that the browser caches it and startup is instant.

## How the scoring works

1. The answer and each guess are embedded into a 384-dimension "meaning vector" using [`all-MiniLM-L6-v2`](https://huggingface.co/Xenova/all-MiniLM-L6-v2) running locally via [Transformers.js](https://huggingface.co/docs/transformers.js) (ONNX Runtime, WASM backend).
2. Similarity is **cosine similarity** between the two vectors.
3. That raw value is remapped onto a friendly 0–100 scale.

Raw cosine similarity is *not* zero for unrelated words — with this model, single words cluster around `0.27–0.40` regardless of meaning. So the mapping is calibrated to the model's real distribution:

| cosine | ~score | example vs `"answer"` |
| --- | --- | --- |
| 0.30 | 0 | `banana`, `table` |
| 0.40 | 15 | `dog` |
| 0.60 | 46 | `response` |
| 0.72 | 65 | `question` |
| 0.85 | 85 | near-synonyms |
| 1.00 | 100 | `answer` (exact match) |

Only the exact answer can score 100 — everything else is capped at 99.

### Tuning

All three knobs live at the top of `game.js`:

```js
const SIM_FLOOR = 0.3;   // cosine that maps to 0  (raise = stricter)
const SIM_CEIL  = 0.95;  // cosine that maps to 100
const SIM_GAMMA = 1.0;   // curve sharpness (raise = only very close words score high)
```

If a different answer word feels too easy or too hard, adjust `SIM_FLOOR` first.

## Changing the answer

```js
const ANSWER = (params.get('word') || 'answer').trim().toLowerCase();
```

Edit that literal, or test alternate words without touching code:

```
http://127.0.0.1:5173/?word=ocean
```

## The ending

Solving the riddle doesn't show a "you win" modal — it fades the board out and reveals a dinner invitation:

```
[ answer. logo ]
you won
a dinner date
happy hour · see you there
```

**Drop the logo at `assets/answer.png`.** It's shown on a dark background, so a PNG with a transparent background works best (an opaque white rectangle will read as a white block).

If the file is missing, the reveal falls back to a typeset `answer.` wordmark, so the page never breaks.

Timing lives in one constant in `game.js`:
- `REVEAL_DELAY` (1400 ms) — how long the winning row stays on screen before the fade

## Files

| File | Purpose |
| --- | --- |
| `index.html` | Markup: board, HUD, loader overlay, reveal scene |
| `styles.css` | Dark theme, closeness colour tiers, animations |
| `game.js` | Model loading, embeddings, scoring, game state |
| `assets/answer.png` | The logo shown in the reveal (drop it in here) |

## Known trade-offs

- **`all-MiniLM-L6-v2` is a sentence model.** It captures *association* more than strict synonymy, so `question` scores above `reply` for the answer `answer`. I benchmarked `bge-small-en-v1.5` too — it had a higher noise floor (`~0.55` for unrelated words) and a narrower spread, so MiniLM gives a better game curve.
- **Word validation is a heuristic.** It filters keyboard mashing (`zzzz`) but won't catch every non-word; junk that slips through simply scores ~0. Swap `looksLikeWord()` for a `Set` lookup if you bundle a dictionary. (The free Datamuse API can't be used here — it sends no CORS headers, so browsers block it.)
- **No rank number.** Contexto shows a rank against a fixed vocabulary (~100k words). Doing that here means embedding the whole vocabulary once, which is a much heavier job. Showing a 0–100 score avoids shipping a word list.
