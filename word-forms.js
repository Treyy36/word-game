/**
 * Inflection collapsing — the one definition of "the same word".
 *
 * Shared by game.js (which collapses guesses, so `answers` cannot be scored
 * twice) and by scripts/build-ranks.mjs (which collapses the vocabulary, so no
 * rank is ever occupied by a word the game would refuse to show). Keeping it in
 * a single module is what stops the two from drifting apart and quietly
 * reintroducing dead ranks.
 *
 * A suffix is only stripped when the result is a real dictionary word, so
 * `bus`, `class` and `news` survive intact while `cities` becomes `city` and
 * `stopped` becomes `stop`.
 */

/** Longest first, so "answers" tries the plural `s` only after `es` fails. */
export const INFLECTIONS = [
  ['ies', 'y'],
  ['ing', ''],
  ['est', ''],
  ['ed', ''],
  ['es', ''],
  ['s', ''],
];

/**
 * The base form of `word`, or `word` unchanged when no rule applies.
 * `dictionary` is a Set of lowercase words; without it nothing is collapsed.
 */
export function canonicalWord(word, dictionary) {
  if (!dictionary || word.length < 4 || word.includes("'") || word.includes('-')) {
    return word;
  }

  for (const [suffix, replacement] of INFLECTIONS) {
    if (!word.endsWith(suffix)) continue;

    const stem = word.slice(0, -suffix.length);
    if (stem.length < 3) continue;

    const candidates = [];
    if (replacement) candidates.push(stem + replacement);
    candidates.push(stem);

    // "stopped" -> "stopp" -> "stop"
    if (stem.at(-1) === stem.at(-2)) candidates.push(stem.slice(0, -1));

    // "hiding" -> "hid" -> "hide"
    candidates.push(`${stem}e`);

    for (const candidate of candidates) {
      if (dictionary.has(candidate)) return candidate;
    }
  }

  return word;
}
