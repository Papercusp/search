import { defineConfig } from 'vitest/config';

// The library's Postgres-backed suite (*.integration.test.ts). Standalone like
// vitest.config.ts (WI-4973): its database comes from src/__testing__/native-pg.ts,
// never from a host's test harness.
export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/**/*.integration.test.ts'],
    exclude: ['node_modules', 'dist'],
    globalSetup: ['./src/__testing__/native-pg.ts'],
    testTimeout: 60_000,
    hookTimeout: 120_000,
  },
});
