/**
 * contexto-ish — guess the secret word, scored by semantic closeness.
 *
 * HOW THE SCORING WORKS
 * ---------------------
 * Every word (the answer and each guess) is turned into a 384-dimension
 * "meaning vector" by a small sentence-transformer model running locally in
 * the browser (all-MiniLM-L6-v2 via Transformers.js / ONNX Runtime WASM).
 * Similarity between two vectors is cosine similarity, which we squash into a
 * friendly 0-100 score. No server, no API key, no network call per guess.
 *
 * TUNING
 * ------
 * Cosine similarity for this model is not zero for unrelated words — single
 * words cluster around 0.27-0.40 even when they have nothing to do with each
 * other. So we remap [SIM_FLOOR, SIM_CEIL] onto [0, 100]: anything at or below
 * the noise floor scores 0, near-synonyms (~0.85) score in the 90s.
 *   - SIM_FLOOR: cosine similarity that maps to 0 (raise to be stricter)
 *   - SIM_CEIL:  cosine similarity that maps to 100
 *   - SIM_GAMMA: curve sharpness (higher = only very close words score high)
 */

const MODEL_LIB = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';
const MODEL_ID = 'Xenova/all-MiniLM-L6-v2';

const SIM_FLOOR = 0.3;
const SIM_CEIL = 0.95;
const SIM_GAMMA = 1.0;

const MAX_WORDS = 3;

/** The secret word. Hard-coded as requested; `?word=foo` overrides it for testing. */
const params = new URLSearchParams(location.search);
const ANSWER = (params.get('word') || 'answer').trim().toLowerCase();

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
  bestScore: document.getElementById('best-score'),
  answerSlots: document.getElementById('answer-slots'),
  loader: document.getElementById('loader'),
  loaderBar: document.getElementById('loader-bar'),
  loaderText: document.getElementById('loader-text'),
  reveal: document.getElementById('reveal'),
};

/* -------------------------------------------------------------------------- */
/* State                                                                      */
/* -------------------------------------------------------------------------- */

let extractor = null;
let answerVector = null;

const vectorCache = new Map();
const guessedWords = new Set();

let guessTotal = 0;
let best = 0;
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

function similarityToScore(similarity) {
  const stretched = (similarity - SIM_FLOOR) / (SIM_CEIL - SIM_FLOOR);
  const clamped = Math.min(1, Math.max(0, stretched));
  return Math.round(Math.pow(clamped, SIM_GAMMA) * 100);
}

function tierOf(score) {
  if (score >= 100) return 'solved';
  if (score >= 80) return 'hot';
  if (score >= 60) return 'warm';
  if (score >= 40) return 'mild';
  if (score >= 22) return 'cool';
  if (score >= 10) return 'cold';
  return 'freezing';
}

function tierMessage(score) {
  if (score >= 80) return 'Scorching. You are basically there. 🔥';
  if (score >= 60) return 'Warm — same neighbourhood.';
  if (score >= 40) return 'Cool. Related, but not yet.';
  if (score >= 22) return 'Cold. Different area of meaning.';
  if (score >= 10) return 'Freezing. Almost nothing in common. 🧊';
  return 'Glacial. Try a completely different direction.';
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

/** Returns an error message, or null when the guess is acceptable. */
function checkFormat(word) {
  if (!word) return 'Type a word first.';
  if (!/^[a-z][a-z' -]*$/.test(word)) return 'Letters, apostrophes and hyphens only.';
  if (word.split(' ').length > MAX_WORDS) return `Keep it to ${MAX_WORDS} words or fewer.`;
  if (word.length > 40) return 'That is too long.';
  return null;
}

/**
 * Cheap offline sanity check that filters keyboard mashing ("asdf", "zzzzqwt")
 * without rejecting real-but-uncommon words. To use a real dictionary instead,
 * swap the body of this function for a Set lookup — the caller stays the same.
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

function createRow(word, score) {
  const row = document.createElement('li');
  row.className = `guess-row ${tierOf(score)}`;
  row.dataset.score = String(score);

  const wordEl = document.createElement('span');
  wordEl.className = 'guess-word';
  wordEl.textContent = word;

  const scoreEl = document.createElement('span');
  scoreEl.className = 'guess-score';
  scoreEl.textContent = String(score);

  const bar = document.createElement('span');
  bar.className = 'guess-bar';
  const fill = document.createElement('span');
  fill.className = 'guess-bar-fill';
  bar.append(fill);

  row.append(wordEl, scoreEl, bar);

  // Animate the bar from 0 → score on the next frame.
  requestAnimationFrame(() => {
    fill.style.width = `${score}%`;
  });

  return row;
}

/** Keeps the board ordered best-first, like Contexto. */
function insertSorted(row, score) {
  const sibling = Array.from(ui.list.children).find(
    (child) => Number(child.dataset.score) < score,
  );

  if (sibling) ui.list.insertBefore(row, sibling);
  else ui.list.append(row);
}

/* -------------------------------------------------------------------------- */
/* Game flow                                                                  */
/* -------------------------------------------------------------------------- */

function registerGuess(word, score) {
  guessedWords.add(word);
  guessTotal += 1;
  ui.guessCount.textContent = String(guessTotal);

  if (score > best) {
    best = score;
    ui.bestScore.textContent = String(score);
    ui.bestScore.classList.add('revealed');
  }

  insertSorted(createRow(word, score), score);

  ui.input.value = '';
  ui.input.focus();

  if (word === ANSWER) {
    finish();
  } else {
    setStatus(tierMessage(score), score >= 60 ? 'hot' : 'muted');
  }
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
 * Fades the board out and the dinner invitation in. Safe to call repeatedly —
 * the reveal only ever plays once per page load.
 */
function showReveal() {
  if (revealed) return;
  revealed = true;

  document.body.classList.add('revealed');
  ui.reveal.hidden = false;

  // Two frames so the browser registers the hidden -> visible change first,
  // otherwise the opacity transition is skipped entirely.
  requestAnimationFrame(() => {
    requestAnimationFrame(() => ui.reveal.classList.add('is-visible'));
  });
}

async function handleGuess(event) {
  event.preventDefault();
  if (gameOver || busy || !answerVector) return;

  const word = normalizeGuess(ui.input.value);

  const formatProblem = checkFormat(word);
  if (formatProblem) {
    setStatus(formatProblem, 'error');
    ui.input.select();
    return;
  }

  if (guessedWords.has(word)) {
    setStatus(`You already tried “${word}”.`, 'error');
    ui.input.select();
    return;
  }

  if (!looksLikeWord(word)) {
    setStatus(`“${word}” does not look like a word. Try a real one.`, 'error');
    ui.input.select();
    return;
  }

  busy = true;
  ui.button.disabled = true;

  try {
    setStatus('Thinking…');

    const vector = await getVector(word);
    const similarity = cosineSimilarity(vector, answerVector);
    // Only the exact answer can score 100.
    const score = word === ANSWER ? 100 : Math.min(99, similarityToScore(similarity));

    registerGuess(word, score);
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

  try {
    extractor = await createExtractor();
    ui.loaderText.textContent = 'Encoding the secret word…';
    answerVector = await getVector(ANSWER);

    ui.loader.hidden = true;
    ui.input.disabled = false;
    ui.button.disabled = false;
    ui.input.focus();

    setStatus(`${ANSWER.length} letters. Start guessing — meaning counts, not spelling.`);
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
