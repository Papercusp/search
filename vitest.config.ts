import { defineConfig } from 'vitest/config';

// WI-4973: standalone config — @papercusp/search is a public, independently
// consumable package (github.com/Papercusp/search); it must build/test with
// only its own declared deps, never routing through the Papercusp-monorepo
// private `@papercusp/test-config` harness. Pure node algorithm tests, no DOM.
export default defineConfig({
  test: {
    environment: 'node',
    exclude: ['node_modules', 'dist'],
    testTimeout: 15_000,
  },
});
