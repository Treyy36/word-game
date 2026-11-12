/**
 * trey-texto — guess the secret word, scored by semantic closeness.
 *
 * HOW THE SCORING WORKS
 * ---------------------
 * Every word (the answer and each guess) is turned into a 384-dimension
 * "meaning vector" by a small sentence-transformer model running locally in
 * the browser (all-MiniLM-L6-v2 via Transformers.js / ONNX Runtime WASM).
 * Similarity between two vectors is cosine similarity, which we convert into a
 * rank: 1 is the answer, and larger numbers are further away in meaning.
 * No server, no API key, no network call per guess.
 *
 * TUNING
 * ------
 * rank = exp(RANK_K * (1 - closeness)), where closeness normalises cosine
 * similarity against SIM_MIN. This is a monotonic distance index — it is NOT a
 * count of how many words are closer, which would require embedding a whole
 * vocabulary up front.
 *   - SIM_MIN: similarity that maps to the largest common ranks
 *   - RANK_K:  curve steepness (higher = ranks climb faster with distance)
 */

const MODEL_LIB = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';
const MODEL_ID = 'Xenova/all-MiniLM-L6-v2';

/** ENABLE word list (~173k words), served from assets/ and loaded once. */
const DICTIONARY_URL = 'assets/words.txt';
/** Frequency-ordered words, used to pick the likely intent behind a typo. */
const FREQUENCY_URL = 'assets/word-frequency.txt';
const DICTIONARY_TIMEOUT_MS = 20000;

/**
 * Rank curve. Similarity maps to a rank where 1 is the answer and larger
 * numbers mean further away. SIM_MIN is the similarity that lands at
 * exp(RANK_K) ≈ 8,100; below it the rank keeps climbing smoothly.
 */
const SIM_MIN = 0.1;
const RANK_K = 9;
const MAX_RANK = 999999;
/** Rank that renders as zero-width bar — anything worse is clamped. */
const BAR_MAX_RANK = 50000;

/** Guesses that get a note instead of a score. Checked before any validation. */
const EASTER_EGGS = new Map([
  ['trey', 'cool guy'],
  ['harper', 'Princess'],
]);

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
  closestRank: document.getElementById('closest-rank'),
  answerSlots: document.getElementById('answer-slots'),
  loader: document.getElementById('loader'),
  loaderBar: document.getElementById('loader-bar'),
  loaderText: document.getElementById('loader-text'),
  reveal: document.getElementById('reveal'),
  plan: document.getElementById('plan'),
  planOpen: document.getElementById('plan-open'),
  planClose: document.getElementById('plan-close'),
  weatherBody: document.getElementById('weather-body'),
  lightbox: document.getElementById('lightbox'),
  lightboxImg: document.getElementById('lightbox-img'),
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

/** Bar width for a rank — logarithmic so mid-range ranks stay distinguishable. */
function closenessPercent(rank) {
  if (rank <= 1) return 100;
  const pct = 100 * (1 - Math.log10(rank) / Math.log10(BAR_MAX_RANK));
  return Math.max(0, Math.min(100, Math.round(pct)));
}

/**
 * Green when close, yellow in the middle, red when far off. Interpolated on a
 * log scale so the colour shifts as smoothly as the bar does.
 */
function tierColor(rank) {
  if (rank <= 1) return '#35d07f';

  const closeness = Math.min(1, Math.max(0, 1 - Math.log10(rank) / Math.log10(BAR_MAX_RANK)));
  const hue = Math.round(120 * closeness); // 120 = green, 60 = yellow, 0 = red
  const lightness = Math.round(46 + 12 * (1 - closeness));

  return `hsl(${hue} 68% ${lightness}%)`;
}

function tierOf(rank) {
  if (rank <= 1) return 'solved';
  if (rank <= 10) return 'hot';
  if (rank <= 100) return 'warm';
  if (rank <= 1000) return 'mild';
  if (rank <= 10000) return 'cool';
  if (rank <= 50000) return 'cold';
  return 'freezing';
}

function tierMessage(rank) {
  if (rank <= 10) return 'Very close — you are basically there.';
  if (rank <= 100) return 'Close. Same neighbourhood.';
  if (rank <= 1000) return 'Same general area of meaning.';
  if (rank <= 10000) return 'Far off — a different area of meaning.';
  if (rank <= 50000) return 'Very far off.';
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

/** Returns an error message, or null when the guess is acceptable. */
function checkFormat(word) {
  if (!word) return 'Type a word first.';
  if (/\s/.test(word)) return 'One word at a time.';
  if (!/^[a-z][a-z' -]*$/.test(word)) return 'Letters, apostrophes and hyphens only.';
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
 * Real dictionary lookup when it loaded, heuristic otherwise. Contractions and
 * hyphenated words skip the lookup: word lists like ENABLE don't carry
 * apostrophes, and they are still a single word.
 */
function isKnownWord(word) {
  if (dictionary && !word.includes("'") && !word.includes('-')) {
    return dictionary.has(word);
  }
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
  guesses.set(word, { rank });
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

  ui.input.value = '';
  ui.input.focus();

  if (word === ANSWER) {
    finish();
  } else if (message) {
    setStatus(message, 'info');
  } else {
    setStatus(tierMessage(rank), rank <= 100 ? 'close' : 'muted');
  }
}

/** Echoes what a repeated guess already scored and points at its row. */
function reportRepeat(word) {
  const previous = guesses.get(word);
  const detail = previous.egg ? previous.egg : `rank ${previous.rank}`;
  setStatus(`You already guessed “${word}” — ${detail}.`, 'info');
  flashRow(word);
  ui.input.value = '';
  ui.input.focus();
}

/** Easter eggs are counted but never ranked — they carry no rank. */
function registerEgg(word, message) {
  guesses.set(word, { egg: message });
  guessTotal += 1;
  ui.guessCount.textContent = String(guessTotal);

  unpinPrevious();
  ui.list.prepend(createEggRow(word, message));

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

  document.documentElement.classList.add('revealed');
  ui.reveal.hidden = false;

  // Two frames so the browser registers the hidden -> visible change first,
  // otherwise the opacity transition is skipped entirely.
  requestAnimationFrame(() => {
    requestAnimationFrame(() => ui.reveal.classList.add('is-visible'));
  });
}

/* -------------------------------------------------------------------------- */
/* Itinerary                                                                  */
/* -------------------------------------------------------------------------- */

const WEATHER = {
  latitude: 36.1627,
  longitude: -86.7816,
  timezone: 'America/Chicago',
};

let weatherRequested = false;

function formatClock(iso) {
  const [hourText, minuteText] = (iso.split('T')[1] ?? '').split(':');
  const hour = Number(hourText);
  const suffix = hour >= 12 ? 'pm' : 'am';
  const hour12 = hour % 12 === 0 ? 12 : hour % 12;
  return `${hour12}:${minuteText} ${suffix}`;
}

function weatherTip(rain) {
  if (rain >= 40) return 'rain is likely — bring an umbrella.';
  return 'cool evening — perfect for a cardigan ;)';
}

function renderWeather(data) {
  const hourly = data?.hourly;
  if (!hourly?.time?.length) throw new Error('weather response missing hourly data');

  // hourly.time[0] is midnight local, so index 17 is the 5:00 pm slot.
  const slot = Math.min(17, hourly.time.length - 1);
  const temp = Math.round(hourly.temperature_2m[slot]);
  const feels = Math.round(hourly.apparent_temperature[slot]);
  const rain = Math.round(hourly.precipitation_probability?.[slot] ?? 0);
  const sunset = data.daily?.sunset?.[0];

  ui.weatherBody.innerHTML = `
    <div class="weather-main">
      <span class="weather-temp">${temp}&deg;</span>
      <div class="weather-side">
        <span class="weather-at">at 5:00 pm</span>
        <span class="weather-feels">feels like ${feels}&deg;</span>
      </div>
    </div>
    <ul class="weather-stats">
      <li><span>rain</span><strong>${rain}%</strong></li>
      ${sunset ? `<li><span>sunset</span><strong>${formatClock(sunset)}</strong></li>` : ''}
    </ul>
    <p class="weather-tip">${weatherTip(rain)}</p>
  `;
}

async function loadWeather() {
  if (weatherRequested) return;
  weatherRequested = true;

  const params = new URLSearchParams({
    latitude: String(WEATHER.latitude),
    longitude: String(WEATHER.longitude),
    hourly: 'temperature_2m,apparent_temperature,precipitation_probability',
    daily: 'sunset',
    temperature_unit: 'fahrenheit',
    timezone: WEATHER.timezone,
    forecast_days: '1',
  });

  try {
    const response = await fetch(`https://api.open-meteo.com/v1/forecast?${params}`);
    if (!response.ok) throw new Error(`weather request failed: ${response.status}`);
    renderWeather(await response.json());
  } catch (error) {
    console.error(error);
    const note = document.createElement('p');
    note.className = 'weather-loading';
    note.textContent = 'could not load the forecast — check your connection.';
    ui.weatherBody.replaceChildren(note);
  }
}

function openPlan() {
  ui.plan.hidden = false;
  ui.plan.scrollTop = 0;
  requestAnimationFrame(() => {
    requestAnimationFrame(() => ui.plan.classList.add('is-visible'));
  });
  loadWeather();
}

function closePlan() {
  ui.plan.classList.remove('is-visible');
  setTimeout(() => {
    ui.plan.hidden = true;
  }, 420);
}

function openLightbox(src, label) {
  ui.lightboxImg.src = src;
  ui.lightboxImg.alt = label ? `${label} menu` : 'menu';
  ui.lightbox.hidden = false;
}

function closeLightbox() {
  ui.lightbox.hidden = true;
  ui.lightboxImg.removeAttribute('src');
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

  if (guesses.has(word)) {
    reportRepeat(word);
    return;
  }

  if (EASTER_EGGS.has(word)) {
    registerEgg(word, EASTER_EGGS.get(word));
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

      if (guesses.has(word)) {
        setStatus(`“${typed}” → “${word}” — already guessed.`, 'info');
        flashRow(word);
        ui.input.value = '';
        ui.input.focus();
        return;
      }

      if (EASTER_EGGS.has(word)) {
        registerEgg(word, EASTER_EGGS.get(word));
        return;
      }
    }

    setStatus('Thinking…');

    const vector = await getVector(word);
    const similarity = cosineSimilarity(vector, answerVector);
    // Only the exact answer can sit at rank 1.
    const rank = word === ANSWER ? 1 : Math.max(2, similarityToRank(similarity));

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

  ui.planOpen.addEventListener('click', openPlan);
  ui.planClose.addEventListener('click', closePlan);
  ui.lightbox.addEventListener('click', closeLightbox);

  ui.plan.addEventListener('click', (event) => {
    const thumb = event.target.closest('.menu-thumb');
    if (thumb) openLightbox(thumb.dataset.full, thumb.dataset.label);
  });

  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape') return;
    if (!ui.lightbox.hidden) closeLightbox();
    else if (!ui.plan.hidden) closePlan();
  });

  // Starts downloading immediately; awaited before the input unlocks.
  const wordListsLoad = loadWordLists();

  try {
    extractor = await createExtractor();
    ui.loaderText.textContent = 'Encoding the secret word…';
    answerVector = await getVector(ANSWER);
    await wordListsLoad;

    ui.loader.hidden = true;
    ui.input.disabled = false;
    ui.button.disabled = false;
    ui.input.focus();

    setStatus(`${ANSWER.length} letters. Start guessing.`);
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
