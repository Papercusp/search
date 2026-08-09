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
 * WHY IT MATTERS, concretely: a card renders `highlight || excerpt`, and
 * nothing in that string says which LEG retrieved the row. A vector-only hit
 * is a row the lexical ranker did not return at all — it was retrieved on
 * meaning — so the terms a reader sees (or does not see) in the excerpt are
 * not the reason it is on screen. Without provenance the reader cannot tell a
 * strong semantic hit from a broken search.
 *
 * ⚠ MEASURED, and it corrects the obvious intuition (2026-08-03, live corpus,
 * query "why did the gate go red", 30 hits): a vector-only hit does NOT
 * generally come back unmarked. 9 of 10 `semantic` hits had <mark> in their
 * highlight, and 2 of 20 `lexical` hits had none. `ts_headline` runs over the
 * DOCUMENT against the tsquery at hydration time, independently of which
 * ranker returned the row — so a turn the BM25 leg ranked too low to return
 * can still contain query terms and get them marked, and a lexical hit can
 * come back unmarked when the headline window lands away from them.
 *
 * So "vector-only" and "no highlight" are DIFFERENT conditions and must not be
 * inferred from one another: this module answers WHICH LEG RETRIEVED IT, and
 * a caller that wants "is the text on screen actually the match" has to test
 * the rendered string for <mark> separately.
 *
 * Deliberately tolerant of ranker RENAMES: the engine's lexical leg is
 * registered as `lexical` (plus `lexical-fresh`, the recency-window candidate
 * leg) since P-013 renamed it from `bm25`/`bm25-fresh`. BOTH spellings still
 * read as lexical here — see `isLexicalRanker` for why that is not a
 * deprecation alias — so neither the rename nor a host's own naming can
 * silently flip every card to "semantic match".
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

/** Ranker labels that mean "term/lexical match". `lexical` is the engine's
 *  current name and `lexical-fresh` its recency-window candidate leg; `bm25`
 *  is the pre-P-013 spelling. Prefix-matched so a `<family>-*` variant stays
 *  lexical without a code change here.
 *
 *  RETAINING `bm25` IS NOT A DEPRECATION ALIAS, which the repo otherwise bans
 *  in alpha. This is a CLASSIFIER over labels this library does not control:
 *  `SearchSource.name` and the ranker label are supplied by the HOST, and
 *  @papercusp/search is published standalone, so a host that legitimately
 *  names its own lexical source `bm25` must not be silently classified
 *  `semantic`. The failure mode is the reason the tolerance exists: a wrong
 *  answer with no error, on the provenance line a user reads to decide whether
 *  a result is a real term match. There is no write path here to keep
 *  bilingual — the engine emits `lexical` only. */
export function isLexicalRanker(ranker: string): boolean {
  return ranker.startsWith('lexical') || ranker.startsWith('bm25');
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
