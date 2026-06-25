import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig({
  base: './',
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: [
        'favicon.svg',
        'favicon-32.png',
        'apple-touch-icon.png',
      ],
      manifest: {
        // Full title for the install dialog; short_name is what shows under
        // the icon on iOS / Android home screens (≤ 12 chars recommended).
        name: '资治通鉴 · 胡三省音注',
        short_name: '资治通鉴',
        description: '《资治通鉴》带胡三省音注的网页阅读器，离线可用。',
        lang: 'zh-Hans',
        start_url: './',
        scope: './',
        display: 'standalone',
        orientation: 'portrait',
        background_color: '#f4ecd8',
        theme_color: '#7a3b2e',
        icons: [
          { src: 'icon-192.png',          sizes: '192x192', type: 'image/png' },
          { src: 'icon-512.png',          sizes: '512x512', type: 'image/png' },
          { src: 'icon-maskable-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
          { src: 'apple-touch-icon.png',  sizes: '180x180', type: 'image/png' },
        ],
      },
      workbox: {
        // App shell only — keep the precache small so first install is quick.
        // The 18+ MB of juan/manifest/lookup JSON is cached lazily on first
        // use via the runtimeCaching rule below, then served offline.
        globPatterns: ['**/*.{js,css,html,svg,png,woff,woff2,ttf}'],
        navigateFallback: 'index.html',
        navigateFallbackDenylist: [/^\/text\//],
        runtimeCaching: [
          {
            // Juan JSON, manifest, lookup index — everything under text/.
            urlPattern: ({ url }) => url.pathname.includes('/text/') && url.pathname.endsWith('.json'),
            handler: 'StaleWhileRevalidate',
            options: {
              cacheName: 'zztj-text-v1',
              expiration: {
                // Whole corpus is finite (~300 juan files + lookup + manifest),
                // plus per-卷 白话导读 guide files under text/guide/.
                maxEntries: 700,
                // Effectively never expire — text doesn't go stale.
                maxAgeSeconds: 60 * 60 * 24 * 365,
              },
              cacheableResponse: { statuses: [0, 200] },
            },
          },
          {
            // Google Fonts CSS — small, may change occasionally.
            urlPattern: ({ url }) => url.origin === 'https://fonts.googleapis.com',
            handler: 'StaleWhileRevalidate',
            options: { cacheName: 'google-fonts-css' },
          },
          {
            // Google Fonts files — immutable; cache long.
            urlPattern: ({ url }) => url.origin === 'https://fonts.gstatic.com',
            handler: 'CacheFirst',
            options: {
              cacheName: 'google-fonts-files',
              expiration: { maxEntries: 30, maxAgeSeconds: 60 * 60 * 24 * 365 },
              cacheableResponse: { statuses: [0, 200] },
            },
          },
        ],
      },
    }),
  ],
  server: {
    port: 5173,
    open: false,
  },
});
