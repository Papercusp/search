import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

// Inside the Papercusp monorepo, run its restricted-hold preflight before any test starts
// (Decision D-012, WI-10005765): this standalone config cannot import @papercusp/test-config,
// which carries that preflight for every other package. A standalone clone of this repo has no
// such file, so it runs none.
function monorepoHostPreflight(): string[] {
  for (let dir = dirname(fileURLToPath(import.meta.url)); ; dir = dirname(dir)) {
    const candidate = join(dir, 'scripts/lib/vitest-host-preflight.mjs');
    if (existsSync(candidate)) return [candidate];
    if (dirname(dir) === dir) return [];
  }
}

// WI-4973: standalone config — @papercusp/search is a public, independently
// consumable package (github.com/Papercusp/search); it must build/test with
// only its own declared deps, never routing through the Papercusp-monorepo
// private `@papercusp/test-config` harness. Pure node algorithm tests, no DOM.
export default defineConfig({
  test: {
    globalSetup: monorepoHostPreflight(),
    environment: 'node',
    // *.integration.test.ts needs Postgres: vitest.integration.config.ts runs it.
    exclude: ['node_modules', 'dist', 'src/**/*.integration.test.ts'],
    testTimeout: 15_000,
  },
});
