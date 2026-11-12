# trey-texto

A small browser word game in the style of [Contexto](https://contexto.me): guess the secret word, and every guess gets a **literal rank** — the number of the 10,000 most common English words that sit closer in *meaning* to the answer, plus one. Pure HTML/CSS/JS: no server, no API keys, nothing to compile at runtime.

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
3. That similarity becomes a **literal rank**: how many of the 10,000 most common English words sit closer to the answer than your guess, plus one.

The rank is a real position, not a curve. `assets/ranks.txt` ships those 10,000 words already sorted by similarity to the answer, so scoring a guess is just finding where it slots in.

| rank | example vs `"answer"` | |
| --- | --- | --- |
| 1 | `answer` | exact match |
| 2 | `question` | one word is closer |
| 3 | `reply` | |
| 5 | `response` | |
| 18 | `ask` | |
| 174 | `solve` | |
| 522 | `dog` | |
| 1,252 | `pizza` | |
| 4,618 | `banana` | |

Only the answer — or an inflection of it — can reach rank 1; every other guess starts at 2 or worse. Guesses *outside* the 10,000 are still embedded in the browser and slotted into the same ordered list, so the count stays honest.

Each new guess is **pinned to the top** of the board so it never gets lost, and the row it displaces drops back into rank order underneath. Repeat guesses don't add a row — they flash the existing one and echo its rank.

### Every rank is reachable

Rank *N* has to name a word the game will actually show. That fails in one way: if two entries are the same base form, guessing either one collapses onto the other, and the loser's rank can never appear — the sequence reads 1, 4, 6, 8 with holes in it.

So the vocabulary holds **one slot per base form**. `answered`, `answers` and `answering` all collapse to `answer` and are dropped from the table, which is why `question` sits at 2 rather than 4. `build-ranks.mjs` asserts the invariant and fails loudly if it ever breaks:

```
Duplicate base forms survived dedupe — dead ranks would come back.
2 entries are not stable base forms, e.g. "posses"
```

That second check is not theoretical: `possessed` collapses to `possess`, which collapses again to `posses`, which is *also* a plural of `posse`. One pass is not enough, so `baseForm()` iterates to a fixed point before anything is written.

### Hand-pinning the top of the list

Ranks 1–30 are the ones a player actually sees and talks about, so they can be overridden by hand. `TOP_WORDS` in `build-ranks.mjs` takes an ordered list that is written to ranks 1..N verbatim; everything below keeps its similarity order.

```js
const TOP_WORDS = ['answer', 'question', 'reply', /* … */];
```

The build rejects a list that doesn't start with the answer, names a word outside the vocabulary, contains a duplicate, or contains anything that isn't already a base form. It ships empty, because collapsing inflections already produces a dense, sensible top 30 — pinning is there if you want to hand-tune the wording.

### The vocabulary

`scripts/build-ranks.mjs` builds the table:

- vocabulary = the top 10,000 entries of `assets/word-frequency.txt` that are also in the ENABLE dictionary, lowercase a–z only, **collapsed to one slot per base form**
- each word is embedded **one at a time** — batching changes the pooled vector for short inputs, and the browser only ever embeds a single word, so single encodes are the only ones that match production
- the list is sorted by similarity to the answer and written to `assets/ranks.txt` (163 KB, ~60 KB gzipped)

```powershell
cd scripts
npm install
node build-ranks.mjs                # answer = "answer", 10,000 words
node build-ranks.mjs banana 20000   # different answer, bigger vocabulary
```

The whole build takes about 10 seconds. A bigger vocabulary means bigger numbers and finer discrimination; 10,000 keeps the table under 200 KB while separating the top of the list cleanly.

### If the table is missing

`?word=` changes the answer, which invalidates every number in the table. `loadRankTable()` reads the header (`#answer=answer size=10000`) and refuses a table built for a different word, logging a warning and falling back to the old exponential curve:

```js
rank = exp(RANK_K * (1 - closeness))   // SIM_MIN = 0.1, RANK_K = 9
```

That is a monotonic distance index rather than a real count, so the magnitudes get fuzzy but the ordering is still correct. Re-run `build-ranks.mjs` for the new answer to get literal ranks back.

### Tuning

```js
const SIM_MIN = 0.1;   // fallback curve: similarity that maps to the largest ranks
const RANK_K  = 9;     // fallback curve: steepness (raise = ranks climb faster)
```

With the table loaded, the thing worth tuning is instead the vocabulary size and the tier boundaries in `tierOf()` / `tierMessage()` — currently the top 0.1%, 1%, 10%, 40% and 80% of the vocabulary.

## Changing the answer

```js
const ANSWER = (params.get('word') || 'answer').trim().toLowerCase();
```

Edit that literal, then rebuild the rank table so the numbers describe the new word:

```powershell
cd scripts
node build-ranks.mjs ocean
```

To try other words without touching code:

```
http://127.0.0.1:5173/?word=ocean
```

That path uses the fallback curve, since the shipped table was built for a different answer.

## The ending

Solving the riddle doesn't show a "you win" modal — it fades the board out and reveals a dinner invitation over a full-bleed photo of the restaurant:

```
you won
a dinner date
answer.
friends · happy hour · see you there
```

`assets/answer.png` is the backdrop. It fills the screen (`background-size: cover`) with a slow 11s drift and a gradient scrim behind the type, so a bright interior shot still reads.

Timing lives in one constant in `game.js`:
- `REVEAL_DELAY` (1400 ms) — how long the winning row stays on screen before the fade

## The itinerary

A `see the plan` button under the invitation opens a scrollable itinerary:

- **timeline** — pickup 4:40–5:00 pm, dinner 5:00–5:45 pm (with a note that happy hour runs 4:15–6:00 so both menus are available), then dessert at home and the evening's plans
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
| `word-forms.js` | Inflection collapsing — shared by the game and the build so they can't drift |
| `scripts/build-ranks.mjs` | Builds `assets/ranks.txt` — the literal rank vocabulary |
| `scripts/pdf-to-images.py` | Renders the menu PDFs to PNGs for the itinerary |
| `assets/answer.png` | Full-bleed backdrop for the reveal |
| `assets/ranks.txt` | 10,000 words sorted by similarity to the answer (the rank table) |
| `assets/words.txt` | ENABLE word list used to reject misspellings |
| `assets/word-frequency.txt` | Frequency-ordered words: typo resolution + rank vocabulary source |
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
- **The vocabulary is 10,000 words, not Contexto's ~100k.** Ranks are literal counts, so they are only as fine-grained as the list behind them: nothing can score worse than 10,001, and the answer's nearest neighbours land at 2–10 rather than 2–50. A 100k table would be ~1.6 MB and take about ten times as long to build, for a game that is usually solved in a dozen guesses.
- **A guess is one plain word.** Whitespace, hyphens, digits, and leading or trailing punctuation are rejected before anything is scored. That is not just tidiness: `answer-`, `answer'` and `answer-answer` used to sail past the old `[a-z' -]` pattern, skip the dictionary entirely and get embedded as near-copies of the answer, so they out-ranked `question`. A single internal apostrophe is still allowed so contractions (`don't`, `o'clock`) work.
- **Plurals and inflections count as one word.** `canonicalWord()` (in `word-forms.js`) strips a common ending (`-s`, `-es`, `-ies`, `-ed`, `-ing`, `-est`) and returns the first stem that is itself a dictionary word, so `answers`, `answered` and `answer` share a single slot on the board and any of them can solve it. The board still shows what she actually typed. `-er`/`-ly` are deliberately *not* stripped, since that would merge `corner` into `corn` and `letter` into `let`. A few unrelated pairs still collapse (`news` → `new`, `means` → `mean`), which is why the rank table is keyed by base form — `news` reports `new`'s rank rather than inventing its own.
