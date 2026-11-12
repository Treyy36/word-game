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
3. That value becomes a **rank**: `1` is the answer, larger numbers are further away.

| cosine | ~rank | example vs `"answer"` |
| --- | --- | --- |
| 1.00 | 1 | `answer` (exact match) |
| 0.85 | ~4 | near-synonyms |
| 0.72 | ~16 | `question` |
| 0.60 | ~54 | `response` |
| 0.40 | ~400 | `dog` |
| 0.30 | ~1,100 | `banana` |
| 0.15 | ~5,000 | unrelated |
| 0.00 | ~22,000 | nothing in common |

Only the exact answer can reach rank 1; every other guess starts at 2 or worse.

Each new guess is **pinned to the top** of the board so it never gets lost, and the row it displaces drops back into rank order underneath. Repeat guesses don't add a row — they flash the existing one and echo its rank.

### What the rank actually is

It is a **distance index**, not a literal position in a vocabulary. A real rank would mean "there are 3,472 words closer than this one", which requires embedding a whole word list up front — impractical in a browser (173k words through WASM would take tens of minutes). `similarityToRank()` instead maps similarity through `exp(RANK_K * (1 - closeness))`, which is monotonic and deterministic: the *ordering* is truthful, and the magnitudes are tuned to feel Contexto-like.

### Tuning

```js
const SIM_MIN = 0.1;   // similarity that maps to the largest common ranks
const RANK_K  = 9;     // curve steepness (raise = ranks climb faster)
```

Raise `RANK_K` or lower `SIM_MIN` if good guesses feel too generous.

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
| `assets/words.txt` | ENABLE word list used to reject misspellings |
| `assets/word-frequency.txt` | Frequency-ordered words, used to resolve typos |
| `assets/*.pdf` | Source menus |
| `assets/*.png` (menus) | Generated menu images used by the itinerary |

## Easter eggs

Guesses listed in `EASTER_EGGS` skip scoring entirely and show a note instead of a number:

```js
const EASTER_EGGS = new Map([
  ['trey', 'cool guy'],
  ['harper', 'Princess'],
]);
```

They are checked before word validation, so they work even for names that aren't in the dictionary. Egg rows are prepended to the board, carry no rank, and don't affect the closest-rank stat.

## Known trade-offs

- **`all-MiniLM-L6-v2` is a sentence model.** It captures *association* more than strict synonymy, so `question` scores above `reply` for the answer `answer`. I benchmarked `bge-small-en-v1.5` too — it had a higher noise floor (`~0.55` for unrelated words) and a narrower spread, so MiniLM gives a better game curve.
- **Spelling is checked, and typos are auto-corrected.** Guesses must appear in `assets/words.txt` (ENABLE, ~173k entries) and be a single word. Anything else is run through a one-edit search — insert, delete, swap or substitute a letter — and the best candidate is used, with the swap announced in the status line (*"cappucino" → "cappuccino"*). Candidates are ranked by real-world frequency from `assets/word-frequency.txt`, because edit distance alone can't distinguish `answer` from `ansae`. If nothing plausible turns up, the guess is rejected. If the lists fail to load, `looksLikeWord()` takes over as a permissive fallback. (The free Datamuse API can't be used here — it sends no CORS headers, so browsers block it.)
- **No rank number.** Contexto shows a rank against a fixed vocabulary (~100k words). Doing that here means embedding the whole vocabulary once, which is a much heavier job. Showing a 0–100 score avoids shipping a word list.
- **Plurals and inflections count as one word.** `canonicalWord()` strips a common ending (`-s`, `-es`, `-ies`, `-ed`, `-ing`, `-est`) and returns the first stem that is itself a dictionary word, so `answers`, `answered` and `answer` share a single slot on the board and any of them can solve it. The board still shows what she actually typed. `-er`/`-ly` are deliberately *not* stripped, since that would merge `corner` into `corn` and `letter` into `let`. A few unrelated pairs still collapse (`news` → `new`, `means` → `mean`), but only for duplicate detection and the win check — scoring and display always use the raw word.
