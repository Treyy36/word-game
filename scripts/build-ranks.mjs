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

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const ASSETS = path.join(ROOT, 'assets');

const MODEL_ID = 'Xenova/all-MiniLM-L6-v2';
const DICTIONARY_FILE = path.join(ASSETS, 'words.txt');
const FREQUENCY_FILE = path.join(ASSETS, 'word-frequency.txt');
const OUTPUT_FILE = path.join(ASSETS, 'ranks.txt');

const ANSWER = (process.argv[2] || 'answer').trim().toLowerCase();
const VOCAB_SIZE = Number(process.argv[3] || 10000);

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
    if (!candidate || seen.has(candidate)) continue;
    if (!/^[a-z]{2,20}$/.test(candidate)) continue;
    if (!dictionary.has(candidate)) continue;

    seen.add(candidate);
    words.push(candidate);
    if (words.length >= VOCAB_SIZE) break;
  }

  // The answer has to be in the vocabulary, or nothing can out-rank it.
  if (!seen.has(ANSWER)) {
    words.pop();
    words.push(ANSWER);
    console.log(`Added "${ANSWER}" to the vocabulary (it was not in the top ${VOCAB_SIZE}).`);
  }

  return words;
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

async function main() {
  console.log(`Answer: "${ANSWER}" · vocabulary: ${VOCAB_SIZE} words`);

  const words = await buildVocabulary();
  console.log(`Vocabulary built: ${words.length} words`);

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

  const order = Array.from(words.keys()).sort((a, b) => similarity[b] - similarity[a]);

  const lines = [
    `#answer=${ANSWER} size=${words.length} model=${MODEL_ID} dtype=q8`,
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

  console.log(`\nWrote ${OUTPUT_FILE} (${words.length} rows)`);
  console.log('Sample ranks:');
  for (const word of ['answer', 'question', 'reply', 'response', 'ask', 'dog', 'banana', 'pizza']) {
    const rank = rankOf(word);
    console.log(`  ${word.padEnd(10)} ${rank === null ? '(not in vocabulary)' : rank}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
