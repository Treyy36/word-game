/**
 * trey-texto — guess the secret word, scored by semantic closeness.
 *
 * HOW THE SCORING WORKS
 * ---------------------
 * Every word is turned into a 384-dimension "meaning vector" by a small
 * sentence-transformer model running locally in the browser
 * (all-MiniLM-L6-v2 via Transformers.js / ONNX Runtime WASM).
 *
 * The rank is a literal count. assets/ranks.txt ships the 10,000 most common
 * English words pre-sorted by cosine similarity to the answer, so rank N means
 * "exactly N-1 vocabulary words are closer to the answer than this one" and the
 * answer itself is rank 1. No server, no API key, no network call per guess.
 *
 * Guesses outside those 10,000 are embedded here and slotted into the same
 * ordered list, so the count stays honest for obscure words too.
 *
 * FALLBACK
 * --------
 * If the table is missing — or was built for a different answer, which happens
 * as soon as you use `?word=` — rank = exp(RANK_K * (1 - closeness)) instead: a
 * monotonic distance index rather than a real count.
 *   - SIM_MIN: similarity that maps to the largest common ranks
 *   - RANK_K:  curve steepness (higher = ranks climb faster with distance)
 */

import { canonicalWord } from './word-forms.js';

const MODEL_LIB = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';
const MODEL_ID = 'Xenova/all-MiniLM-L6-v2';

/** ENABLE word list (~173k words), served from assets/ and loaded once. */
const DICTIONARY_URL = 'assets/words.txt';
/** Frequency-ordered words, used to pick the likely intent behind a typo. */
const FREQUENCY_URL = 'assets/word-frequency.txt';
/**
 * The vocabulary the ranks are counted against: 10,000 common words sorted by
 * similarity to the answer. Built by scripts/build-ranks.mjs.
 */
const RANKS_URL = 'assets/ranks.txt';
const DICTIONARY_TIMEOUT_MS = 20000;

/**
 * Rank curve, used only when the literal rank table is unavailable. Similarity
 * maps to a rank where 1 is the answer. SIM_MIN is the similarity that lands at
 * exp(RANK_K) ≈ 8,100; below it the rank keeps climbing smoothly.
 */
const SIM_MIN = 0.1;
const RANK_K = 9;
const MAX_RANK = 999999;
/** Rank that renders as a zero-width bar when the fallback curve is in use. */
const BAR_MAX_RANK = 50000;

/** Guesses that get a note instead of a score. Checked before any validation. */
const EASTER_EGGS = new Map([
  ['trey', 'cool guy'],
  ['harper', 'Princess'],
  ['rat', 'Berkley'],
  ['berkley', 'rat'],
  ['baby', ';)'],
]);

/** The secret word. Hard-coded as requested; `?word=foo` overrides it for testing. */
const params = new URLSearchParams(location.search);
const ANSWER = (params.get('word') || 'tulip').trim().toLowerCase();

/* -------------------------------------------------------------------------- */
/* DOM                                                                        */
/* -------------------------------------------------------------------------- */

const ui = {
  form: document.getElementById('guess-form'),
  input: document.getElementById('guess-input'),
  button: document.getElementById('guess-button'),
  status: document.getElementById('status'),
  list: document.getElementById('guess-list'),
  guessCount: document.getElementById('guess-count'),
  closestRank: document.getElementById('closest-rank'),
  answerSlots: document.getElementById('answer-slots'),
  loader: document.getElementById('loader'),
  loaderBar: document.getElementById('loader-bar'),
  loaderText: document.getElementById('loader-text'),
  reveal: document.getElementById('reveal'),
  note: document.getElementById('note'),
  noteOpen: document.getElementById('note-open'),
  noteClose: document.getElementById('note-close'),
};

/* -------------------------------------------------------------------------- */
/* State                                                                      */
/* -------------------------------------------------------------------------- */

let extractor = null;
let answerVector = null;

const vectorCache = new Map();

/** word -> { rank } for scored guesses, { egg } for easter eggs. */
const guesses = new Map();

/** Populated by loadWordLists(); null means fall back to the heuristic. */
let dictionary = null;
/** word -> position in the frequency list (0 = most common). */
let frequencyRank = null;
/**
 * Populated by loadRankTable(): { answer, size, byWord, scores } where scores
 * is every vocabulary similarity in descending order. Null means the shipped
 * table is missing or describes a different answer.
 */
let rankTable = null;

let guessTotal = 0;
let bestRank = Number.POSITIVE_INFINITY;
let gameOver = false;
let busy = false;
let revealed = false;

/** Let the winning row land before the board fades away. */
const REVEAL_DELAY = 1400;

/* -------------------------------------------------------------------------- */
/* Model bootstrap                                                            */
/* -------------------------------------------------------------------------- */

function onModelProgress(info) {
  if (info.status === 'progress' && info.total) {
    const pct = Math.round((info.loaded / info.total) * 100);
    ui.loaderBar.style.width = `${pct}%`;
    ui.loaderText.textContent = `Downloading model… ${pct}%`;
  } else if (info.status === 'done') {
    ui.loaderText.textContent = 'Almost there…';
  }
}

async function createExtractor() {
  const { pipeline, env } = await import(MODEL_LIB);

  // Everything is fetched from the CDN; nothing is bundled locally.
  env.allowLocalModels = false;

  // Single-threaded WASM so we don't depend on cross-origin isolation headers.
  if (env.backends?.onnx?.wasm) {
    env.backends.onnx.wasm.numThreads = 1;
  }

  const options = { progress_callback: onModelProgress };

  try {
    return await pipeline('feature-extraction', MODEL_ID, { ...options, dtype: 'q8' });
  } catch (error) {
    // Some mirrors only ship full-precision weights — retry once before failing.
    console.warn('Quantised weights unavailable, retrying with defaults.', error);
    return await pipeline('feature-extraction', MODEL_ID, options);
  }
}

/** Embedding for a single word, memoised for the session. */
async function getVector(word) {
  const key = word.toLowerCase();
  if (vectorCache.has(key)) return vectorCache.get(key);

  const output = await extractor(key, { pooling: 'mean', normalize: true });
  const vector = Float32Array.from(output.data);
  vectorCache.set(key, vector);
  return vector;
}

/* -------------------------------------------------------------------------- */
/* Scoring                                                                    */
/* -------------------------------------------------------------------------- */

function cosineSimilarity(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  return denominator === 0 ? 0 : dot / denominator;
}

function similarityToRank(similarity) {
  const closeness = (similarity - SIM_MIN) / (1 - SIM_MIN);
  const rank = Math.exp(RANK_K * (1 - closeness));
  return Math.max(1, Math.min(MAX_RANK, Math.round(rank)));
}

/**
 * Inserts a similarity into the shipped ordered list and returns its position.
 * Everything at or above index `low` is closer, so `low + 1` is the rank.
 */
function rankForSimilarity(similarity) {
  const scores = rankTable.scores;
  let low = 0;
  let high = scores.length;

  while (low < high) {
    const mid = (low + high) >> 1;
    if (scores[mid] > similarity) low = mid + 1;
    else high = mid;
  }

  return low + 1;
}

/**
 * The literal rank of a word we shipped a score for, else null. The table is
 * keyed by canonical form, so inflections find their base word's slot.
 */
function vocabularyRank(key) {
  if (!rankTable) return null;
  return rankTable.byWord.get(key) ?? null;
}

/**
 * The rank for one guess: its exact position among the shipped vocabulary if we
 * have one, otherwise its position among the same list by measured similarity.
 * Falls back to the exponential curve if no table is loaded at all.
 */
async function scoreWord(word, key) {
  const exact = vocabularyRank(key);
  if (exact !== null) return exact;

  const vector = await getVector(word);
  const similarity = cosineSimilarity(vector, answerVector);

  return rankTable ? rankForSimilarity(similarity) : similarityToRank(similarity);
}

/** Worst rank that still draws a bar — the vocabulary size when we have one. */
function barMaxRank() {
  return rankTable ? rankTable.size : BAR_MAX_RANK;
}

/** Bar width for a rank — logarithmic so mid-range ranks stay distinguishable. */
function closenessPercent(rank) {
  if (rank <= 1) return 100;
  const pct = 100 * (1 - Math.log10(rank) / Math.log10(barMaxRank()));
  return Math.max(0, Math.min(100, Math.round(pct)));
}

/**
 * Green when close, yellow in the middle, red when far off. Interpolated on a
 * log scale so the colour shifts as smoothly as the bar does.
 */
function tierColor(rank) {
  if (rank <= 1) return '#35d07f';

  const closeness = Math.min(1, Math.max(0, 1 - Math.log10(rank) / Math.log10(barMaxRank())));
  const hue = Math.round(120 * closeness); // 120 = green, 60 = yellow, 0 = red
  const lightness = Math.round(46 + 12 * (1 - closeness));

  return `hsl(${hue} 68% ${lightness}%)`;
}

/**
 * Tier boundaries. With 10,000 words to count against, "far off" has to start
 * much earlier than it did on the old 999,999 ceiling — these are the top 0.1%,
 * 1%, 10%, 40% and 80% of the vocabulary.
 */
function tierOf(rank) {
  if (rank <= 1) return 'solved';
  if (rank <= 10) return 'hot';
  if (rank <= 100) return 'warm';
  if (rank <= 1000) return 'mild';
  if (rank <= 4000) return 'cool';
  if (rank <= 8000) return 'cold';
  return 'freezing';
}

function tierMessage(rank) {
  if (rank <= 10) return 'Very close — you are basically there.';
  if (rank <= 100) return 'Close. Same neighbourhood.';
  if (rank <= 1000) return 'Same general area of meaning.';
  if (rank <= 4000) return 'Far off — a different area of meaning.';
  if (rank <= 8000) return 'Very far off.';
  return 'Nothing in common. Try a completely different direction.';
}

/* -------------------------------------------------------------------------- */
/* Input helpers                                                              */
/* -------------------------------------------------------------------------- */

function normalizeGuess(value) {
  return value
    .toLowerCase()
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Returns an error message, or null when the guess is acceptable.
 *
 * A guess has to be one plain word. A hyphen means two words, and a dangling
 * apostrophe or hyphen (`answer'`, `answer-`) slips past a loose pattern and
 * reaches the model as a near-copy of the answer, scoring absurdly close.
 */
function checkFormat(word) {
  if (!word) return 'Type a word first.';
  if (/\s/.test(word)) return 'One word at a time.';
  if (word.includes('-')) return 'One word at a time — no hyphens.';
  if (!/^[a-z]+(?:'[a-z]+)?$/.test(word)) {
    return 'Letters only, with at most one apostrophe in the middle.';
  }
  if (word.length > 40) return 'That is too long.';
  return null;
}

/** Loads the dictionary and the frequency list. Safe to call once. */
async function loadWordLists() {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), DICTIONARY_TIMEOUT_MS);

  try {
    const [dictionaryResponse, frequencyResponse] = await Promise.all([
      fetch(DICTIONARY_URL, { signal: controller.signal }),
      fetch(FREQUENCY_URL, { signal: controller.signal }),
    ]);

    if (!dictionaryResponse.ok) {
      throw new Error(`dictionary request failed: ${dictionaryResponse.status}`);
    }

    const dictionaryText = await dictionaryResponse.text();
    dictionary = new Set(
      dictionaryText
        .split('\n')
        .map((entry) => entry.trim().toLowerCase())
        .filter(Boolean),
    );

    if (frequencyResponse.ok) {
      const frequencyText = await frequencyResponse.text();
      frequencyRank = new Map();
      frequencyText.split('\n').forEach((line, index) => {
        const entry = line.trim().toLowerCase().split(/\s+/)[0];
        if (entry && /^[a-z]+$/.test(entry) && !frequencyRank.has(entry)) {
          frequencyRank.set(entry, index);
        }
      });
    }

    console.info(
      `Word lists loaded: ${dictionary.size} words, ${frequencyRank?.size ?? 0} ranked.`,
    );
  } catch (error) {
    // Never block the game on this — fall back to the heuristic below.
    console.warn('Word lists unavailable, falling back to heuristic word check.', error);
    dictionary = null;
    frequencyRank = null;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Loads the precomputed rank table: one "word similarity" pair per line, in
 * descending order of similarity to the answer, so a word's rank is just its
 * line number. The header records which answer it was built for — if that
 * doesn't match ANSWER (somebody used `?word=`), every number in it is wrong
 * and we discard it rather than lie.
 */
async function loadRankTable() {
  try {
    const response = await fetch(RANKS_URL);
    if (!response.ok) throw new Error(`rank table request failed: ${response.status}`);

    const [header, ...rows] = (await response.text()).split('\n');
    const meta = /answer=(\S+)\s+size=(\d+)/.exec(header);
    if (!meta) throw new Error('rank table header is missing its answer/size fields');

    const [, tableAnswer] = meta;
    if (tableAnswer !== ANSWER) {
      console.warn(
        `Rank table was built for "${tableAnswer}", not "${ANSWER}" — using the fallback curve.`,
      );
      return;
    }

    const byWord = new Map();
    const scores = [];

    for (const row of rows) {
      const [word, score] = row.trim().split(/\s+/);
      if (!word || score === undefined) continue;

      // First occurrence wins — the list is closest-first, so a word that got
      // listed twice should keep its better rank.
      if (!byWord.has(word)) byWord.set(word, scores.length + 1);
      scores.push(Number(score));
    }

    rankTable = { answer: tableAnswer, size: scores.length, byWord, scores: Float32Array.from(scores) };
    console.info(`Rank table loaded: ${scores.length} words scored against "${tableAnswer}".`);
  } catch (error) {
    // The exponential curve keeps the game playable without it.
    console.warn('Rank table unavailable, falling back to the similarity curve.', error);
    rankTable = null;
  }
}

/**
 * Real dictionary lookup when it loaded, heuristic otherwise. Contractions skip
 * the lookup — word lists like ENABLE don't carry apostrophes.
 */
function isKnownWord(word) {
  if (dictionary && !word.includes("'")) return dictionary.has(word);
  return looksLikeWord(word);
}

const ALPHABET = 'abcdefghijklmnopqrstuvwxyz';

/** Every real word one edit away from `word` (insert/delete/swap/substitute). */
function oneEditCandidates(word) {
  const candidates = new Set();

  for (let i = 0; i < word.length; i += 1) {
    candidates.add(word.slice(0, i) + word.slice(i + 1));
  }

  for (let i = 0; i < word.length - 1; i += 1) {
    candidates.add(word.slice(0, i) + word[i + 1] + word[i] + word.slice(i + 2));
  }

  for (let i = 0; i < word.length; i += 1) {
    for (const letter of ALPHABET) {
      if (letter !== word[i]) {
        candidates.add(word.slice(0, i) + letter + word.slice(i + 1));
      }
    }
  }

  for (let i = 0; i <= word.length; i += 1) {
    for (const letter of ALPHABET) {
      candidates.add(word.slice(0, i) + letter + word.slice(i));
    }
  }

  return [...candidates].filter((candidate) => dictionary.has(candidate));
}

/**
 * Suggests the intended word for a typo. One-edit candidates are ranked by how
 * common they are — without that, "answe" is just as plausible as "ansae".
 * Returns null when nothing stands out, so the guess is rejected instead.
 */
function findCorrection(word) {
  if (!dictionary) return null;

  const candidates = oneEditCandidates(word);
  if (candidates.length === 0) return null;

  if (frequencyRank) {
    let best = null;
    let bestRank = Infinity;

    for (const candidate of candidates) {
      const rank = frequencyRank.get(candidate);
      if (rank !== undefined && rank < bestRank) {
        bestRank = rank;
        best = candidate;
      }
    }

    if (best) return best;
  }

  // Only one possible word and it is uncommon — still a safe bet.
  return candidates.length === 1 ? candidates[0] : null;
}

/**
 * Cheap fallback that filters keyboard mashing ("asdf", "zzzzqwt") if the
 * dictionary fails to download.
 */
function looksLikeWord(word) {
  const letters = word.replace(/[^a-z]/g, '');
  if (letters.length < 2) return false;
  if (!/[aeiouy]/.test(letters)) return false;
  if (/(.)\1{2,}/.test(letters)) return false;
  if (/^[^aeiouy]{5,}$/.test(letters)) return false;
  return true;
}

/* -------------------------------------------------------------------------- */
/* Rendering                                                                  */
/* -------------------------------------------------------------------------- */

function setStatus(message, tone = 'muted') {
  ui.status.textContent = message;
  ui.status.dataset.tone = tone;
}

function createRow(word, rank) {
  const row = document.createElement('li');
  row.className = `guess-row ${tierOf(rank)}`;
  row.dataset.rank = String(rank);
  row.dataset.word = word;
  row.style.setProperty('--tier', tierColor(rank));

  const wordEl = document.createElement('span');
  wordEl.className = 'guess-word';
  wordEl.textContent = word;

  const rankEl = document.createElement('span');
  rankEl.className = 'guess-rank';
  rankEl.textContent = String(rank);

  const bar = document.createElement('span');
  bar.className = 'guess-bar';
  const fill = document.createElement('span');
  fill.className = 'guess-bar-fill';
  bar.append(fill);

  row.append(wordEl, rankEl, bar);

  // Animate the bar to this rank's closeness on the next frame.
  const width = closenessPercent(rank);
  requestAnimationFrame(() => {
    fill.style.width = `${width}%`;
  });

  return row;
}

/** Keeps the board ordered closest-first, so rank 1 sits at the top. */
function insertSorted(row, rank) {
  const sibling = Array.from(ui.list.children).find(
    (child) => Number(child.dataset.rank) > rank,
  );

  if (sibling) ui.list.insertBefore(row, sibling);
  else ui.list.append(row);
}

/**
 * Files the pinned row back into rank order. Called just before a new guess
 * takes the top slot, so only ever one guess is pinned.
 */
function unpinPrevious() {
  const previous = ui.list.querySelector('.guess-row.latest');
  if (!previous) return;

  previous.classList.remove('latest');
  insertSorted(previous, Number(previous.dataset.rank));
}

/** A guess with no score, used for the easter eggs. */
function createEggRow(word, message) {
  const row = document.createElement('li');
  row.className = 'guess-row egg';
  row.dataset.word = word;

  const wordEl = document.createElement('span');
  wordEl.className = 'guess-word';
  wordEl.textContent = word;

  const messageEl = document.createElement('span');
  messageEl.className = 'guess-egg';
  messageEl.textContent = message;

  row.append(wordEl, messageEl);
  return row;
}

/* -------------------------------------------------------------------------- */
/* Game flow                                                                  */
/* -------------------------------------------------------------------------- */

function registerGuess(word, rank, message = null) {
  guesses.set(canonicalWord(word, dictionary), { rank, word });
  guessTotal += 1;
  ui.guessCount.textContent = String(guessTotal);

  if (rank < bestRank) {
    bestRank = rank;
    ui.closestRank.textContent = String(rank);
    ui.closestRank.classList.add('revealed');
  }

  // The newest guess stays pinned on top so it never gets buried in the list;
  // the row it displaces drops back into rank order underneath.
  unpinPrevious();
  const row = createRow(word, rank);
  row.classList.add('latest');
  ui.list.prepend(row);

  // After the row lands, so the saved board is the one on screen.
  saveProgress();

  ui.input.value = '';
  ui.input.focus();

  if (rank === 1) {
    finish();
  } else if (message) {
    setStatus(message, 'info');
  } else {
    setStatus(tierMessage(rank), rank <= 100 ? 'close' : 'muted');
  }
}

/** Echoes what a repeated guess already scored and points at its row. */
function reportRepeat(key, typed = null) {
  const previous = guesses.get(key);
  const detail = previous.egg ? previous.egg : `rank ${previous.rank}`;
  setStatus(
    typed
      ? `“${typed}” → “${previous.word}” — you already guessed that (${detail}).`
      : `You already guessed “${previous.word}” — ${detail}.`,
    'info',
  );
  flashRow(previous.word);
  ui.input.value = '';
  ui.input.focus();
}

/** Easter eggs are counted but never ranked — they carry no rank. */
function registerEgg(word, message) {
  guesses.set(canonicalWord(word, dictionary), { egg: message, word });
  guessTotal += 1;
  ui.guessCount.textContent = String(guessTotal);

  unpinPrevious();
  ui.list.prepend(createEggRow(word, message));
  saveProgress();

  ui.input.value = '';
  ui.input.focus();
  setStatus('');
}

/** Draws the eye to an existing row when someone repeats a guess. */
function flashRow(word) {
  const row = Array.from(ui.list.children).find((child) => child.dataset.word === word);
  if (!row) return;

  row.classList.remove('flash');
  void row.offsetWidth; // force reflow so the animation can restart
  row.classList.add('flash');
  row.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

/* -------------------------------------------------------------------------- */
/* Saved progress                                                             */
/* -------------------------------------------------------------------------- */

/**
 * Every guess is mirrored into localStorage under the answer's own key, so a
 * reload — including the pull-to-refresh you get from flicking back up to the
 * input on a phone — resumes the round instead of resetting it. A new word
 * starts clean because the key changes with it.
 */
const PROGRESS_KEY = `trey-texto:v1:${ANSWER}`;

function saveProgress() {
  // Saved in on-screen order, so a reload puts every row back exactly where the
  // player left it — including the pinned newest row.
  const entries = [...ui.list.children]
    .map((row) => {
      const { word } = row.dataset;
      const saved = guesses.get(canonicalWord(word, dictionary));
      if (!saved) return null;

      return {
        word,
        rank: saved.rank ?? null,
        egg: saved.egg ?? null,
        latest: row.classList.contains('latest'),
      };
    })
    .filter(Boolean);

  try {
    localStorage.setItem(PROGRESS_KEY, JSON.stringify(entries));
  } catch {
    // Private browsing or a full quota: the game still plays, it just won't
    // survive a reload.
  }
}

function readProgress() {
  try {
    const saved = JSON.parse(localStorage.getItem(PROGRESS_KEY) ?? 'null');
    return Array.isArray(saved) ? saved : null;
  } catch {
    return null;
  }
}

/**
 * Rebuilds the board from saved progress, in the order it was saved in. Returns
 * true when the saved round had already been won.
 */
function restoreProgress() {
  const saved = readProgress();
  if (!saved) return false;

  let won = false;

  for (const entry of saved) {
    if (!entry || typeof entry.word !== 'string') continue;

    const { word } = entry;
    const key = canonicalWord(word, dictionary);
    let row;

    if (typeof entry.rank === 'number') {
      guesses.set(key, { rank: entry.rank, word });
      if (entry.rank < bestRank) bestRank = entry.rank;
      if (entry.rank === 1) won = true;
      row = createRow(word, entry.rank);
    } else if (typeof entry.egg === 'string') {
      guesses.set(key, { egg: entry.egg, word });
      row = createEggRow(word, entry.egg);
    } else {
      continue;
    }

    if (entry.latest) row.classList.add('latest');
    ui.list.append(row);
    guessTotal += 1;
  }

  ui.guessCount.textContent = String(guessTotal);

  if (bestRank !== Number.POSITIVE_INFINITY) {
    ui.closestRank.textContent = String(bestRank);
    ui.closestRank.classList.add('revealed');
  }

  return won;
}

function revealAnswer() {
  ui.answerSlots.textContent = ANSWER.toUpperCase();
  ui.answerSlots.classList.add('revealed');
}

function lockInput() {
  gameOver = true;
  ui.input.disabled = true;
  ui.button.disabled = true;
}

function finish() {
  lockInput();
  revealAnswer();
  setStatus('Solved! 🎉', 'success');
  setTimeout(showReveal, REVEAL_DELAY);
}

/**
 * Fades the board out and the bouquet in. Safe to call repeatedly — the reveal
 * only ever plays once per page load.
 */
function showReveal() {
  if (revealed) return;
  revealed = true;

  document.documentElement.classList.add('revealed');
  ui.reveal.hidden = false;

  // Two frames so the browser registers the hidden -> visible change first,
  // otherwise the opacity transition is skipped entirely.
  requestAnimationFrame(() => {
    requestAnimationFrame(() => ui.reveal.classList.add('is-visible'));
  });
}

/* -------------------------------------------------------------------------- */
/* The card that came with the bouquet                                        */
/* -------------------------------------------------------------------------- */

function openNote() {
  ui.note.hidden = false;
  ui.note.scrollTop = 0;
  requestAnimationFrame(() => {
    requestAnimationFrame(() => ui.note.classList.add('is-visible'));
  });
}

function closeNote() {
  ui.note.classList.remove('is-visible');
  setTimeout(() => {
    ui.note.hidden = true;
  }, 420);
}

async function handleGuess(event) {
  event.preventDefault();
  if (gameOver || busy || !answerVector) return;

  let word = normalizeGuess(ui.input.value);

  const formatProblem = checkFormat(word);
  if (formatProblem) {
    setStatus(formatProblem, 'error');
    ui.input.select();
    return;
  }

  if (EASTER_EGGS.has(word)) {
    const eggKey = canonicalWord(word, dictionary);
    if (guesses.has(eggKey)) reportRepeat(eggKey);
    else registerEgg(word, EASTER_EGGS.get(word));
    return;
  }

  busy = true;
  ui.button.disabled = true;

  try {
    let typed = null;

    if (!isKnownWord(word)) {
      const correction = findCorrection(word);

      if (!correction) {
        setStatus(`“${word}” isn't a word — check the spelling.`, 'error');
        ui.input.select();
        return;
      }

      typed = word;
      word = correction;
    }

    // "answers", "answered" and "answer" are one word, so they share a slot
    // in the guess list and any of them can solve it.
    const key = canonicalWord(word, dictionary);

    if (guesses.has(key)) {
      reportRepeat(key, typed);
      return;
    }

    if (EASTER_EGGS.has(key)) {
      registerEgg(key, EASTER_EGGS.get(key));
      return;
    }

    setStatus('Thinking…');

    // Only the answer, or an inflection of it, sits at rank 1.
    const answerKey = canonicalWord(ANSWER, dictionary);
    const rank = key === answerKey ? 1 : Math.max(2, await scoreWord(word, key));

    registerGuess(
      word,
      rank,
      typed ? `“${typed}” → “${word}” · ${tierMessage(rank)}` : null,
    );
  } catch (error) {
    console.error(error);
    setStatus('Something went wrong scoring that word. Try again.', 'error');
  } finally {
    busy = false;
    ui.button.disabled = gameOver;
  }
}

/* -------------------------------------------------------------------------- */
/* Start                                                                      */
/* -------------------------------------------------------------------------- */

async function start() {
  ui.answerSlots.textContent = '?'.repeat(ANSWER.length);

  ui.form.addEventListener('submit', handleGuess);

  ui.noteOpen.addEventListener('click', openNote);
  ui.noteClose.addEventListener('click', closeNote);

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !ui.note.hidden) closeNote();
  });

  // Starts downloading immediately; awaited before the input unlocks.
  const wordListsLoad = loadWordLists();
  const rankTableLoad = loadRankTable();

  try {
    extractor = await createExtractor();
    ui.loaderText.textContent = 'Encoding the secret word…';
    answerVector = await getVector(ANSWER);
    await Promise.all([wordListsLoad, rankTableLoad]);

    ui.loader.hidden = true;

    // A reload mid-round resumes the board; a reload after the win puts the
    // bouquet back rather than leaving a locked, empty game.
    if (restoreProgress()) {
      revealAnswer();
      lockInput();
      setStatus('Solved! 🎉', 'success');
      setTimeout(showReveal, 600);
      return;
    }

    ui.input.disabled = false;
    ui.button.disabled = false;
    ui.input.focus();

    setStatus(
      guessTotal > 0
        ? `Back where you left off — ${guessTotal} guesses so far.`
        : `${ANSWER.length} letters. Start guessing.`,
    );
  } catch (error) {
    console.error(error);
    ui.loader.hidden = true;
    setStatus(
      'Could not load the language model. Check your internet connection, then reload.',
      'error',
    );
  }
}

start();
