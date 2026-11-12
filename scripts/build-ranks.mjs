/**
 * Builds assets/ranks.txt — the literal rank of every word in a fixed
 * vocabulary, measured against the secret answer.
 *
 * trey-texto used to guess at a rank with a curve (exp of normalised cosine).
 * This script instead answers the question properly: of the N most common
 * English words, how many are closer to the answer than this guess? That count
 * + 1 is the rank, so `answer` is 1 and `question` is whatever it really is.
 *
 * Vocabulary = the frequency list (assets/word-frequency.txt, most common
 * first) intersected with the ENABLE dictionary (assets/words.txt) so we never
 * rank noise like "http" or "www". Only lowercase a-z runs of 2-20 letters.
 *
 * One slot per base form. Without that, `answered`, `answers` and `answering`
 * each eat a rank the game can never show — typing any of them collapses onto
 * `answer` and wins at rank 1. Collapsing here means rank N always names a
 * different word from rank N-1, so there are no gaps to fall into.
 *
 * TOP_WORDS then pins the first ranks by hand. Ranks 1-30 are the ones a player
 * actually sees, so they are curated once and held fixed; everything below them
 * is whatever the similarity ordering says.
 *
 * Words are embedded one at a time on purpose. Batching changes the pooled
 * vector for short inputs, and the browser only ever embeds a single word, so
 * single encodes are the only ones that agree with production.
 *
 * Usage:
 *   node build-ranks.mjs                # answer = "answer", 10k words
 *   node build-ranks.mjs banana 20000   # different answer / bigger vocabulary
 *
 * Re-run this whenever you change ANSWER in game.js — otherwise the shipped
 * table describes the wrong word and the game falls back to the curve.
 */

import { pipeline, env } from '@huggingface/transformers';
import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { canonicalWord } from '../word-forms.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const ASSETS = path.join(ROOT, 'assets');

const MODEL_ID = 'Xenova/all-MiniLM-L6-v2';
const DICTIONARY_FILE = path.join(ASSETS, 'words.txt');
const FREQUENCY_FILE = path.join(ASSETS, 'word-frequency.txt');
const OUTPUT_FILE = path.join(ASSETS, 'ranks.txt');

const ANSWER = (process.argv[2] || 'answer').trim().toLowerCase();
const VOCAB_SIZE = Number(process.argv[3] || 10000);

/**
 * Ranks 1-30, pinned by hand. These are the numbers a player actually sees and
 * talks about, so they are worth curating: every entry is a distinct base form
 * the game will display, and none of them are an inflection of another entry.
 * Leave this empty to let the similarity ordering decide everything.
 *
 * The first entry must be the answer.
 */
const TOP_WORDS = [];

if (!/^[a-z]+$/.test(ANSWER)) {
  console.error(`Answer must be plain lowercase letters, got "${ANSWER}".`);
  process.exit(1);
}

env.allowLocalModels = false;

/* -------------------------------------------------------------------------- */
/* Vocabulary                                                                 */
/* -------------------------------------------------------------------------- */

async function buildVocabulary() {
  const [frequencyText, dictionaryText] = await Promise.all([
    readFile(FREQUENCY_FILE, 'utf8'),
    readFile(DICTIONARY_FILE, 'utf8'),
  ]);

  const dictionary = new Set(
    dictionaryText
      .split('\n')
      .map((line) => line.trim().toLowerCase())
      .filter(Boolean),
  );

  const words = [];
  const seen = new Set();

  for (const line of frequencyText.split('\n')) {
    const candidate = line.trim().toLowerCase().split(/\s+/)[0];
    if (!candidate) continue;
    if (!/^[a-z]{2,20}$/.test(candidate)) continue;
    if (!dictionary.has(candidate)) continue;

    const key = baseForm(candidate, dictionary);
    if (key === null || seen.has(key)) continue;

    seen.add(key);
    words.push(key);
    if (words.length >= VOCAB_SIZE) break;
  }

  // The answer has to be in the vocabulary, or nothing can out-rank it.
  const answerKey = canonicalWord(ANSWER, dictionary);
  if (!seen.has(answerKey)) {
    words.pop();
    words.push(answerKey);
    seen.add(answerKey);
    console.log(`Added "${answerKey}" to the vocabulary (it was not in the top ${VOCAB_SIZE}).`);
  }

  return { words, dictionary };
}

/**
 * Reorders the similarity ranking so TOP_WORDS occupy ranks 1..N verbatim.
 * Everything else keeps its relative order and slides in behind them.
 */
function applyPins(order, words, dictionary) {
  if (TOP_WORDS.length === 0) return order;

  const answerKey = canonicalWord(ANSWER, dictionary);
  if (TOP_WORDS[0] !== answerKey) {
    throw new Error(`TOP_WORDS must start with the answer ("${answerKey}"), got "${TOP_WORDS[0]}"`);
  }

  const indexOf = new Map(words.map((word, index) => [word, index]));
  const pinned = new Set();

  for (const word of TOP_WORDS) {
    // These two checks are what actually guarantee "no dead slots": a pinned
    // word must exist, and no two pinned words may be the same base form.
    if (!words.includes(word)) {
      throw new Error(`TOP_WORDS entry "${word}" is not in the vocabulary`);
    }
    if (canonicalWord(word, dictionary) !== word) {
      throw new Error(`TOP_WORDS entry "${word}" is not a base form`);
    }
    if (pinned.has(word)) {
      throw new Error(`TOP_WORDS repeats "${word}"`);
    }
    pinned.add(word);
  }

  const head = TOP_WORDS.map((word) => indexOf.get(word));
  return [...head, ...order.filter((index) => !pinned.has(words[index]))];
}

/* -------------------------------------------------------------------------- */
/* Embedding                                                                  */
/* -------------------------------------------------------------------------- */

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) sum += a[i] * b[i];
  return sum;
}

function normalize(vector) {
  let norm = 0;
  for (const value of vector) norm += value * value;
  norm = Math.sqrt(norm);
  return norm === 0 ? vector : vector.map((value) => value / norm);
}

/**
 * canonicalWord() to a fixed point. One pass is not enough: `possessed` becomes
 * `possess`, which then becomes `posses`, which collapses again to `posse`. The
 * shipped table has to hold stable words, because that is exactly what lets the
 * game map a guess onto a rank without ever missing.
 *
 * Returns null for words that never settle, so they are dropped rather than
 * shipped in a state the game would score inconsistently.
 */
function baseForm(word, dictionary) {
  let current = word;
  for (let pass = 0; pass < 6; pass += 1) {
    const next = canonicalWord(current, dictionary);
    if (next === current) return current;
    current = next;
  }
  return canonicalWord(current, dictionary) === current ? current : null;
}

async function main() {
  console.log(`Answer: "${ANSWER}" · vocabulary: ${VOCAB_SIZE} words`);

  const { words, dictionary } = await buildVocabulary();
  console.log(`Vocabulary built: ${words.length} distinct words (inflections collapsed)`);

  console.log('Loading model…');
  let extractor;
  try {
    extractor = await pipeline('feature-extraction', MODEL_ID, { dtype: 'q8' });
  } catch (error) {
    console.warn('Quantised weights unavailable, retrying with defaults.', error.message);
    extractor = await pipeline('feature-extraction', MODEL_ID);
  }

  const embed = async (word) => {
    const output = await extractor(word, { pooling: 'mean', normalize: true });
    return normalize(Float32Array.from(output.data));
  };

  const answerVector = await embed(ANSWER);

  console.log('Embedding vocabulary…');
  const similarity = new Float32Array(words.length);
  const started = Date.now();

  for (let i = 0; i < words.length; i += 1) {
    const vector = await embed(words[i]);
    similarity[i] = dot(vector, answerVector);

    if (i % 250 === 0 || i === words.length - 1) {
      const seconds = (Date.now() - started) / 1000;
      const rate = (i + 1) / seconds;
      process.stdout.write(
        `\r  ${i + 1}/${words.length}  ${rate.toFixed(0)} words/s  ${(similarity[i]).toFixed(3)}    `,
      );
    }
  }
  process.stdout.write('\n');

  const computed = Array.from(words.keys()).sort((a, b) => similarity[b] - similarity[a]);
  const order = applyPins(computed, words, dictionary);

  if (words[order[0]] !== canonicalWord(ANSWER, dictionary)) {
    throw new Error(`Rank 1 came out as "${words[order[0]]}", not the answer.`);
  }

  // The whole point of collapsing inflections: if two ranks were the same base
  // form, or an entry collapsed to something else, the game would rewrite one
  // of them to rank 1 and leave a hole in the sequence. Fail the build instead.
  const distinct = new Set(words);
  if (distinct.size !== words.length) {
    throw new Error('Duplicate base forms survived dedupe — dead ranks would come back.');
  }
  const unstable = order.filter((index) => canonicalWord(words[index], dictionary) !== words[index]);
  if (unstable.length > 0) {
    throw new Error(
      `${unstable.length} entries are not stable base forms, e.g. "${words[unstable[0]]}"`,
    );
  }

  console.log('\nClosest 30 by similarity alone:');
  computed.slice(0, 30).forEach((index, rank) => {
    console.log(`  ${String(rank + 1).padStart(2)}  ${words[index]}`);
  });

  const lines = [
    `#answer=${ANSWER} size=${order.length} model=${MODEL_ID} dtype=q8`,
    // Line 1 is rank 1, so the rank is just the line number. The top entry is
    // pinned to a clean 1.000000 so float noise can never displace the answer.
    ...order.map((index, rank) =>
      `${words[index]} ${rank === 0 ? '1.000000' : similarity[index].toFixed(6)}`,
    ),
  ];

  await writeFile(OUTPUT_FILE, `${lines.join('\n')}\n`, 'utf8');

  const rankOf = (word) => {
    const index = words.indexOf(word);
    return index < 0 ? null : order.indexOf(index) + 1;
  };

  console.log(`\nWrote ${OUTPUT_FILE} (${order.length} rows)`);
  console.log('\nFinal top 30:');
  order.slice(0, 30).forEach((index, rank) => {
    console.log(`  ${String(rank + 1).padStart(2)}  ${words[index]}`);
  });

  console.log('\nSample ranks:');
  for (const word of ['answer', 'question', 'reply', 'response', 'ask', 'dog', 'banana', 'pizza']) {
    const rank = rankOf(word);
    console.log(`  ${word.padEnd(10)} ${rank === null ? '(not in vocabulary)' : rank}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
