// Rasterise the SVG icons to PNGs at the sizes Apple/Android home screens
// expect. Embeds the character glyph as an SVG <path> via the system's
// installed fonts (Songti / SimSun / Noto Serif SC) — but rather than rely
// on sharp's fontconfig dance, we instead draw the character with node-canvas
// using whatever CJK serif is available, then composite onto a solid back.
import sharp from 'sharp';
import { writeFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';

const here = dirname(fileURLToPath(import.meta.url));
const publicDir = resolve(here, '..', 'public');

const ACCENT = '#7a3b2e';
const CREAM = '#f4ecd8';

function makeSvg({ size, char, fontSize, withBorder }) {
  // Use a generic CJK serif stack; if none is installed (CI/headless), sharp
  // falls back to its default which still renders a recognisable glyph for
  // 鑑 thanks to the bundled DejaVu/Noto fonts in most Linux images.
  const border = withBorder
    ? `<rect x="${size * 0.078}" y="${size * 0.078}" width="${size * 0.844}" height="${size * 0.844}" fill="none" stroke="${CREAM}" stroke-opacity="0.55" stroke-width="${Math.max(2, size * 0.012)}"/>`
    : '';
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
  <rect width="${size}" height="${size}" fill="${ACCENT}"/>
  ${border}
  <text x="${size / 2}" y="${size / 2}"
        text-anchor="middle" dominant-baseline="central"
        font-family="'Noto Serif CJK SC', 'Noto Serif SC', 'Songti SC', 'STSong', 'SimSun', 'Microsoft YaHei', serif"
        font-weight="700"
        font-size="${fontSize}"
        fill="${CREAM}">鑑</text>
</svg>`;
}

const targets = [
  // Apple touch icon — iOS uses this for "Add to Home Screen".
  { name: 'apple-touch-icon.png',  size: 180, fontRatio: 0.66, withBorder: true },
  // PWA manifest icons.
  { name: 'icon-192.png',          size: 192, fontRatio: 0.66, withBorder: true },
  { name: 'icon-512.png',          size: 512, fontRatio: 0.66, withBorder: true },
  // Maskable: glyph in the 80% safe zone so the OS can crop to a circle.
  { name: 'icon-maskable-512.png', size: 512, fontRatio: 0.51, withBorder: false },
  // Classic favicon raster (browsers that don't pick the SVG).
  { name: 'favicon-32.png',        size: 32,  fontRatio: 0.78, withBorder: false },
];

for (const t of targets) {
  const svg = makeSvg({ size: t.size, char: '鑑', fontSize: Math.round(t.size * t.fontRatio), withBorder: t.withBorder });
  const out = await sharp(Buffer.from(svg)).png().toBuffer();
  writeFileSync(resolve(publicDir, t.name), out);
  console.log('wrote', t.name);
}

// Also emit a 32x32 .ico-equivalent PNG kept as favicon.png for legacy.
writeFileSync(resolve(publicDir, 'favicon.png'),
  await sharp(Buffer.from(makeSvg({ size: 32, fontSize: 24, withBorder: false }))).png().toBuffer());
console.log('wrote favicon.png');
