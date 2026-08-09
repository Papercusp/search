import { describe, expect, it } from 'vitest';
import { hitProvenance, isLexicalRanker, isSemanticRanker, isVectorOnly } from './provenance';

/** Minimal hit shape the classifier reads. */
const hit = (rankers: string[], rankerScores?: Record<string, number>) =>
  ({ rankers, ...(rankerScores ? { rankerScores } : {}) }) as Parameters<typeof hitProvenance>[0];

describe('ranker family predicates', () => {
  it('reads the engine’s current lexical names AND the P-013 rename', () => {
    // If this ever regresses, EVERY lexical hit silently renders as a
    // "semantic match" — the exact inversion P-004 exists to prevent.
    expect(isLexicalRanker('lexical')).toBe(true);
    expect(isLexicalRanker('lexical-fresh')).toBe(true);
    expect(isLexicalRanker('lexical')).toBe(true);
    expect(isLexicalRanker('embeddings')).toBe(false);
  });

  it('reads the vector names', () => {
    expect(isSemanticRanker('embeddings')).toBe(true);
    expect(isSemanticRanker('semantic')).toBe(true);
    expect(isSemanticRanker('lexical')).toBe(false);
  });
});

describe('hitProvenance', () => {
  it('classifies a vector-only hit as semantic and carries the cosine', () => {
    expect(hitProvenance(hit(['embeddings'], { embeddings: 0.6142 }))).toEqual({
      matchedBy: 'semantic',
      semanticScore: 0.6142,
    });
  });

  it('classifies a term-only hit as lexical and carries ts_rank_cd', () => {
    expect(hitProvenance(hit(['lexical'], { lexical: 0.0731 }))).toEqual({
      matchedBy: 'lexical',
      lexicalScore: 0.0731,
    });
  });

  it('classifies a hit both legs returned as both, with both native scores', () => {
    expect(hitProvenance(hit(['lexical', 'embeddings'], { lexical: 0.0731, embeddings: 0.6142 }))).toEqual({
      matchedBy: 'both',
      lexicalScore: 0.0731,
      semanticScore: 0.6142,
    });
  });

  it('takes the BEST score across sibling lexical legs (lexical + lexical-fresh)', () => {
    // Both legs are ts_rank_cd over the same row, so the stronger one is the
    // row's lexical strength — not the last one iterated.
    expect(
      hitProvenance(hit(['lexical', 'lexical-fresh'], { lexical: 0.02, 'lexical-fresh': 0.09 })),
    ).toEqual({ matchedBy: 'lexical', lexicalScore: 0.09 });
    expect(
      hitProvenance(hit(['lexical', 'lexical-fresh'], { lexical: 0.09, 'lexical-fresh': 0.02 })),
    ).toEqual({ matchedBy: 'lexical', lexicalScore: 0.09 });
  });

  it('reads the UNION of rankers and rankerScores — either alone can be absent', () => {
    // runFullTextSearch sets rankerScores but the row may carry no `rankers`.
    expect(hitProvenance({ rankerScores: { lexical: 0.05 } } as never)).toEqual({
      matchedBy: 'lexical',
      lexicalScore: 0.05,
    });
    // A fused hit whose native scores were not retained still classifies.
    expect(hitProvenance(hit(['embeddings']))).toEqual({ matchedBy: 'semantic' });
  });

  it('returns unknown — never a guess — for a hit with no attribution', () => {
    expect(hitProvenance(hit([]))).toEqual({ matchedBy: 'unknown' });
    expect(hitProvenance({} as never)).toEqual({ matchedBy: 'unknown' });
  });

  it('omits a score key entirely rather than emitting undefined', () => {
    // The value crosses a JSON boundary; `semanticScore: undefined` and an
    // absent key serialise the same, but the type must not claim a number.
    const p = hitProvenance(hit(['lexical'], { lexical: 0.1 }));
    expect('semanticScore' in p).toBe(false);
  });
});

describe('isVectorOnly', () => {
  it('is true only for a hit no lexical leg returned', () => {
    expect(isVectorOnly(hit(['embeddings'], { embeddings: 0.7 }))).toBe(true);
    expect(isVectorOnly(hit(['lexical', 'embeddings'], { lexical: 0.05, embeddings: 0.7 }))).toBe(false);
    expect(isVectorOnly(hit(['lexical'], { lexical: 0.05 }))).toBe(false);
  });

  it('is FALSE for unknown — absent attribution is not evidence of a vector match', () => {
    // Mislabelling a real lexical hit "semantic match" is the worse error:
    // it makes a correct result read as a fallback.
    expect(isVectorOnly(hit([]))).toBe(false);
  });
});
