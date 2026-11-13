# trey-texto

A small browser word game in the style of [Contexto](https://contexto.me): guess the secret word, and every guess gets a **literal rank** — the number of the 10,000 most common English words that sit closer in *meaning* to the answer, plus one. Pure HTML/CSS/JS: no server, no API keys, nothing to compile at runtime.

The answer is hard-coded to `"tulip"` in `game.js`.

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

| rank | example vs `"tulip"` | |
| --- | --- | --- |
| 1 | `tulip` | exact match |
| 2 | `flower` | one word is closer |
| 3 | `bloom` | |
| 5 | `petal` | |
| 7 | `bouquet` | |
| 30 | `blush` | last hand-pinned rank |
| 248 | `pizza` | |
| 801 | `dog` | |
| 4,177 | `question` | |

Only the answer — or an inflection of it — can reach rank 1; every other guess starts at 2 or worse. Guesses *outside* the 10,000 are still embedded in the browser and slotted into the same ordered list, so the count stays honest.

Each new guess is **pinned to the top** of the board so it never gets lost, and the row it displaces drops back into rank order underneath. Repeat guesses don't add a row — they flash the existing one and echo its rank.

### Reloads don't reset the round

Every guess is mirrored into `localStorage` under `trey-texto:v1:<answer>`, so a reload rebuilds the board exactly as it was — same rows, same order, same pinned guess, same counts. That is what saves a phone player: flicking back up to the input can trigger a pull-to-refresh, and without this it silently threw the round away. A finished round comes back with the bouquet already on screen.

The key contains the answer, so changing the word starts a clean board and `?word=` keeps its own progress. Storage failures (private mode, full quota) are swallowed — the game plays fine, it just won't survive a reload.

`overscroll-behavior-y: none` on `html`/`body` stops the pull-to-refresh gesture itself where browsers support it (Chrome, Firefox, iOS 16+); older Safari falls back to the saved-progress restore.

### Every rank is reachable

Rank *N* has to name a word the game will actually show. That fails in one way: if two entries are the same base form, guessing either one collapses onto the other, and the loser's rank can never appear — the sequence reads 1, 4, 6, 8 with holes in it.

So the vocabulary holds **one slot per base form**: `tulips` collapses to `tulip` and is dropped from the table, so it can never eat a rank the board is unable to show. `build-ranks.mjs` asserts the invariant and fails loudly if it ever breaks:

```
Duplicate base forms survived dedupe — dead ranks would come back.
2 entries are not stable base forms, e.g. "posses"
```

That second check is not theoretical: `possessed` collapses to `possess`, which collapses again to `posses`, which is *also* a plural of `posse`. One pass is not enough, so `baseForm()` iterates to a fixed point before anything is written.

### Hand-pinning the top of the list

Ranks 1–30 are the ones a player actually sees and talks about, so they are pinned by hand, **per answer**. `TOP_WORD_PINS` in `build-ranks.mjs` maps an answer to the ordered list written to ranks 1..N verbatim; everything below keeps its similarity order. An answer with no entry keeps the raw similarity ordering.

```js
const TOP_WORD_PINS = {
  tulip: ['tulip', 'flower', 'bloom', /* … */ 'blush'],
};
```

The pins exist because the model is weak on bare nouns: sorted by similarity alone, `tulip`'s neighbours come out as `tummy`, `tut`, `turd`, `tux`, and a slur lands at rank 30. Pinning puts the words a player would actually try — `flower`, `rose`, `bouquet`, `vase` — where they belong.

The build rejects a list that doesn't start with the answer, names a word outside the vocabulary, contains a duplicate, or contains anything that isn't already a base form.

### The vocabulary

`scripts/build-ranks.mjs` builds the table:

- vocabulary = the top 10,000 entries of `assets/word-frequency.txt` that are also in the ENABLE dictionary, lowercase a–z only, **collapsed to one slot per base form**
- each word is embedded **one at a time** — batching changes the pooled vector for short inputs, and the browser only ever embeds a single word, so single encodes are the only ones that match production
- the list is sorted by similarity to the answer and written to `assets/ranks.txt` (163 KB, ~60 KB gzipped)

```powershell
cd scripts
npm install
node build-ranks.mjs                # answer = "answer", 10,000 words
node build-ranks.mjs tulip          # the shipped answer
node build-ranks.mjs tulip 20000    # same answer, bigger vocabulary
```

The whole build takes about 10 seconds. A bigger vocabulary means bigger numbers and finer discrimination; 10,000 keeps the table under 200 KB while separating the top of the list cleanly.

### If the table is missing

`?word=` changes the answer, which invalidates every number in the table. `loadRankTable()` reads the header (`#answer=tulip size=10000`) and refuses a table built for a different word, logging a warning and falling back to the old exponential curve:

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
const ANSWER = (params.get('word') || 'tulip').trim().toLowerCase();
```

Edit that literal, then rebuild the rank table so the numbers describe the new word:

```powershell
cd scripts
node build-ranks.mjs rose
```

If the new answer deserves a curated top 30, add it to `TOP_WORD_PINS` first; without an entry the top ranks come straight from the similarity ordering, which on a bare noun can be nonsense.

To try other words without touching code:

```
http://127.0.0.1:5173/?word=ocean
```

That path uses the fallback curve, since the shipped table was built for a different answer.

## The ending

Solving the riddle doesn't show a "you win" modal — it fades the board out and reveals the prize over a full-bleed illustration:

```
you won
a bouquet of tulips
tulip.
fresh cut · wrapped in paper · for you
```

`assets/tulips.svg` is the backdrop. It fills the screen (`background-size: cover`) with the same slow 11s drift, and the scrim is a dark wash centred on the words rather than a uniform dim, so the petals keep their colour while the type stays legible.

Timing lives in one constant in `game.js`:
- `REVEAL_DELAY` (1400 ms) — how long the winning row stays on screen before the fade

## The card

The `read the card` button opens the note that came with the bouquet: a cream card on a dark backdrop with a short message and a signature. It is plain markup (`#note` in `index.html`, `.note-card` in `styles.css`) — no assets, no network.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | Markup: board, HUD, loader overlay, reveal scene, the card |
| `styles.css` | Dark theme, closeness colour tiers, animations |
| `game.js` | Model loading, embeddings, scoring, game state, the reveal |
| `word-forms.js` | Inflection collapsing — shared by the game and the build so they can't drift |
| `scripts/build-ranks.mjs` | Builds `assets/ranks.txt` — the literal rank vocabulary |
| `assets/tulips.svg` | Full-bleed bouquet backdrop for the reveal |
| `assets/ranks.txt` | 10,000 words sorted by similarity to the answer (the rank table) |
| `assets/words.txt` | ENABLE word list used to reject misspellings |
| `assets/word-frequency.txt` | Frequency-ordered words: typo resolution + rank vocabulary source |
| `assets/answer.png`, `assets/*.pdf`, `assets/*.png` (menus) | Leftovers from the previous dinner-date reward. Nothing in the current build references them. |

## Easter eggs

Guesses listed in `EASTER_EGGS` skip scoring entirely and show a note instead of a number:

```js
const EASTER_EGGS = new Map([
  ['trey', 'cool guy'],
  ['harper', 'Princess'],
  ['rat', 'Berkley'],
  ['berkley', 'rat'],
  ['baby', ';)'],
]);
```

They are checked before word validation, so they work even for names that aren't in the dictionary. Egg rows are prepended to the board, carry no rank, and don't affect the closest-rank stat.

## Known trade-offs

- **`all-MiniLM-L6-v2` is a sentence model.** It captures *association* more than strict synonymy, and it is noticeably weak on bare nouns: sorted by similarity alone, `tulip`'s neighbours are `tummy`, `tut`, `turd`, `tux` and `oval`, with `flower` not arriving until rank 26. That is what `TOP_WORD_PINS` exists to paper over. I benchmarked `bge-small-en-v1.5` too — it had a higher noise floor (`~0.55` for unrelated words) and a narrower spread, so MiniLM gives a better game curve.
- **Spelling is checked, and typos are auto-corrected.** Guesses must appear in `assets/words.txt` (ENABLE, ~173k entries) and be a single word. Anything else is run through a one-edit search — insert, delete, swap or substitute a letter — and the best candidate is used, with the swap announced in the status line (*"cappucino" → "cappuccino"*). Candidates are ranked by real-world frequency from `assets/word-frequency.txt`, because edit distance alone can't distinguish `answer` from `ansae`. If nothing plausible turns up, the guess is rejected. If the lists fail to load, `looksLikeWord()` takes over as a permissive fallback. (The free Datamuse API can't be used here — it sends no CORS headers, so browsers block it.)
- **The vocabulary is 10,000 words, not Contexto's ~100k.** Ranks are literal counts, so they are only as fine-grained as the list behind them: nothing can score worse than 10,001, and the answer's nearest neighbours land at 2–10 rather than 2–50. A 100k table would be ~1.6 MB and take about ten times as long to build, for a game that is usually solved in a dozen guesses.
- **A guess is one plain word.** Whitespace, hyphens, digits, and leading or trailing punctuation are rejected before anything is scored. That is not just tidiness: `answer-`, `answer'` and `answer-answer` used to sail past the old `[a-z' -]` pattern, skip the dictionary entirely and get embedded as near-copies of the answer, so they out-ranked `question`. A single internal apostrophe is still allowed so contractions (`don't`, `o'clock`) work.
- **Plurals and inflections count as one word.** `canonicalWord()` (in `word-forms.js`) strips a common ending (`-s`, `-es`, `-ies`, `-ed`, `-ing`, `-est`) and returns the first stem that is itself a dictionary word, so `answers`, `answered` and `answer` share a single slot on the board and any of them can solve it. The board still shows what she actually typed. `-er`/`-ly` are deliberately *not* stripped, since that would merge `corner` into `corn` and `letter` into `let`. A few unrelated pairs still collapse (`news` → `new`, `means` → `mean`), which is why the rank table is keyed by base form — `news` reports `new`'s rank rather than inventing its own.
