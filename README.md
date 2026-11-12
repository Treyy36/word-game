# trey-texto

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

## Deploying

Live on GitHub Pages: **https://treyy36.github.io/word-game/**

Everything is static and every path is relative, so it works from a repo subpath with no build step. To redeploy after a change:

```bash
git add -A && git commit -m "..." && git push
```

Pages rebuilds automatically in about 30 seconds. `.nojekyll` keeps Jekyll from touching the files.

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

Solving the riddle doesn't show a "you win" modal — it fades the board out and reveals a dinner invitation over a full-bleed photo of the restaurant:

```
you won
a dinner date
answer.
happy hour · see you there
```

`assets/answer.png` is the backdrop. It fills the screen (`background-size: cover`) with a slow 11s drift and a gradient scrim behind the type, so a bright interior shot still reads.

Timing lives in one constant in `game.js`:
- `REVEAL_DELAY` (1400 ms) — how long the winning row stays on screen before the fade

## The itinerary

A `see the plan` button under the invitation opens a scrollable itinerary:

- **timeline** — pickup 4:50–5:00 pm, dinner 5:00–5:45 pm (with a note that happy hour runs 4:15–6:00 so both menus are available), then dessert at home and the evening's plans
- **the weather** — live forecast for 5:00 pm from [Open-Meteo](https://open-meteo.com) (free, no API key), plus rain chance and sunset, with a custom line
- **my picks** — the highlighted menu items, with the highlight colour named so they're easy to find on the PDFs
- **the menus** — tap a thumbnail for a full-screen view

### Changing the location

The forecast uses fixed coordinates in `game.js`:

```js
const WEATHER = {
  latitude: 36.1627,
  longitude: -86.7816,
  timezone: 'America/Chicago',
};
```

Weather is fetched lazily the first time the itinerary opens, and fails gracefully to a text note if the network is down.

### Menu assets

The itinerary shows PNGs, not PDFs — browsers (iOS Safari especially) handle PDFs inconsistently. `assets/website+menu.pdf` and `assets/HH+2020.pdf` are the sources; regenerate the images with:

```bash
python scripts/pdf-to-images.py
```

That writes `assets/menu-1.png`, `assets/menu-2.png` and `assets/happy-hour-1.png` at 150 DPI. Requires `python -m pip install pymupdf`.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | Markup: board, HUD, loader overlay, reveal scene, itinerary |
| `styles.css` | Dark theme, closeness colour tiers, animations |
| `game.js` | Model loading, embeddings, scoring, game state, weather |
| `scripts/pdf-to-images.py` | Renders the menu PDFs to PNGs for the itinerary |
| `assets/answer.png` | Full-bleed backdrop for the reveal |
| `assets/*.pdf` | Source menus |
| `assets/*.png` (menus) | Generated menu images used by the itinerary |

## Known trade-offs

- **`all-MiniLM-L6-v2` is a sentence model.** It captures *association* more than strict synonymy, so `question` scores above `reply` for the answer `answer`. I benchmarked `bge-small-en-v1.5` too — it had a higher noise floor (`~0.55` for unrelated words) and a narrower spread, so MiniLM gives a better game curve.
- **Word validation is a heuristic.** It filters keyboard mashing (`zzzz`) but won't catch every non-word; junk that slips through simply scores ~0. Swap `looksLikeWord()` for a `Set` lookup if you bundle a dictionary. (The free Datamuse API can't be used here — it sends no CORS headers, so browsers block it.)
- **No rank number.** Contexto shows a rank against a fixed vocabulary (~100k words). Doing that here means embedding the whole vocabulary once, which is a much heavier job. Showing a 0–100 score avoids shipping a word list.
