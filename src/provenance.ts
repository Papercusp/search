/**
 * provenance.ts — WHY did this hit match? (plan
 * semantic-search-fingerprint-coverage-2026-08-03 P-004)
 *
 * A fused hit carries `rankers` (which legs contributed) and `rankerScores`
 * (their PRE-fusion native scores). Both are engine-internal vocabulary, and
 * every UI that wants to say "semantic match" was re-deriving the same
 * classification from those raw fields — each one hard-coding the ranker
 * names, in a different place, over a network boundary. This module is the
 * one place that knows them.
 *
 * WHY IT MATTERS, concretely: a vector-only hit has no matched TERMS, so
 * `ts_headline` has nothing to wrap in <mark> and returns the head of the
 * document unmarked. A card that renders that string looks like it is showing
 * you the match — it is showing you the first 200 characters of a turn, which
 * is exactly as informative as showing a random 200. The reader cannot tell a
 * strong semantic hit from a bug. Naming the provenance is what makes the
 * excerpt honest.
 *
 * Deliberately tolerant of ranker RENAMES: the lexical leg is registered as
 * `bm25` today (plus `bm25-fresh`, the recency-window candidate leg) and
 * P-013 of the same plan renames the public surface to `lexical`. Both read
 * as lexical here, so that rename cannot silently flip every card to
 * "semantic match".
 *
 * Pure — no PG, no host types, unit-tested standalone.
 */
import type { SearchHit } from './types';

/** Which ranker family (or families) produced a hit. `unknown` = the hit
 *  carries no ranker attribution at all — a host-constructed hit, or one that
 *  never went through fusion. Callers must render `unknown` as "no claim
 *  made", never as either of the two real answers. */
export type MatchProvenance = 'lexical' | 'semantic' | 'both' | 'unknown';

export interface HitProvenance {
  matchedBy: MatchProvenance;
  /** Best lexical native score (ts_rank_cd units), when a lexical leg hit. */
  lexicalScore?: number;
  /** Embedding native score (cosine similarity), when the vector leg hit. */
  semanticScore?: number;
}

/** Ranker labels that mean "term/lexical match". `bm25` is the engine's
 *  current name, `bm25-fresh` its recency-window candidate leg, `lexical` the
 *  P-013 rename. Prefix-matched so a future `bm25-*` variant stays lexical. */
export function isLexicalRanker(ranker: string): boolean {
  return ranker.startsWith('bm25') || ranker.startsWith('lexical');
}

/** Ranker labels that mean "vector/semantic match". */
export function isSemanticRanker(ranker: string): boolean {
  return ranker.startsWith('embedding') || ranker.startsWith('semantic');
}

/**
 * Classify a hit's match provenance.
 *
 * Reads the UNION of `rankers` and `rankerScores` keys rather than either
 * alone: `runHybridSearch` populates both, but `runFullTextSearch` sets only
 * `rankerScores`, and a host may construct a hit with neither. Taking the
 * union means a hit is never classified `unknown` while it plainly carries
 * the answer in the other field.
 */
export function hitProvenance(
  hit: Pick<SearchHit, 'rankers' | 'rankerScores'> & { rankers?: string[] },
): HitProvenance {
  const scores = hit.rankerScores ?? {};
  const names = new Set<string>([...(hit.rankers ?? []), ...Object.keys(scores)]);

  let lexicalScore: number | undefined;
  let semanticScore: number | undefined;
  for (const name of names) {
    const score = scores[name];
    if (isLexicalRanker(name)) {
      // Several lexical legs can hit the same row (bm25 + bm25-fresh); they
      // share units, so the best one is the row's lexical strength.
      if (typeof score === 'number' && (lexicalScore === undefined || score > lexicalScore)) {
        lexicalScore = score;
      }
    } else if (isSemanticRanker(name) && typeof score === 'number') {
      semanticScore = score;
    }
  }

  const hasLexical = [...names].some(isLexicalRanker);
  const hasSemantic = [...names].some(isSemanticRanker);
  const matchedBy: MatchProvenance =
    hasLexical && hasSemantic ? 'both' : hasLexical ? 'lexical' : hasSemantic ? 'semantic' : 'unknown';

  return {
    matchedBy,
    ...(lexicalScore === undefined ? {} : { lexicalScore }),
    ...(semanticScore === undefined ? {} : { semanticScore }),
  };
}

/**
 * True when a hit has NO lexical contribution — i.e. nothing in it matched a
 * query TERM, so any highlight/excerpt shown for it is the head of the
 * document rather than the match. The predicate a card uses to decide whether
 * to label the excerpt instead of presenting it as a match.
 *
 * `unknown` returns false: absent attribution is not evidence of a
 * vector-only match, and mislabelling a lexical hit "semantic" is the worse
 * error (it makes a correct result look like a fallback).
 */
export function isVectorOnly(hit: Pick<SearchHit, 'rankers' | 'rankerScores'>): boolean {
  return hitProvenance(hit).matchedBy === 'semantic';
}
