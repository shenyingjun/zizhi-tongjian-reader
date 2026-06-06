// Headless render verification.
import { chromium } from 'playwright';

const URL = process.env.READER_URL || 'http://localhost:5173/';

const TRADITIONAL_MARKERS = ['資', '諸侯', '為'];
const SIMPLIFIED_MARKERS = ['资', '诸侯', '为'];

(async () => {
  const browser = await chromium.launch({ headless: true });
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  page.on('console', msg => {
    if (msg.type() === 'error') console.log('[page error]', msg.text());
  });
  page.on('pageerror', err => console.log('[uncaught]', err.message));

  await page.goto(URL, { waitUntil: 'networkidle' });
  await page.waitForSelector('.paragraph', { timeout: 10000 });

  const title = await page.locator('.reader-title').textContent();
  const sidebarItems = await page.locator('.dynasty-toggle').count();
  const paragraphs = await page.locator('.paragraph').count();
  const huMarkers = await page.locator('.hu-marker').count();
  const body = (await page.locator('.reader-body').textContent()) || '';

  const tradHits = TRADITIONAL_MARKERS.filter(c => body.includes(c));
  const simpHits = SIMPLIFIED_MARKERS.filter(c => body.includes(c));

  console.log('--- Reader render check ---');
  console.log('title:', title?.trim().slice(0, 80));
  console.log('dynasty groups:', sidebarItems);
  console.log('paragraphs:', paragraphs);
  console.log('hu markers:', huMarkers);
  console.log('simplified markers found:', simpHits);
  console.log('traditional markers (should be empty):', tradHits);
  console.log('body sample:', body.slice(0, 120).replace(/\s+/g, ''));

  await page.locator('.hu-marker').first().click();
  const noteText = await page.locator('.hu-note').first().textContent();
  console.log('first hu-note text:', noteText?.slice(0, 60));

  await page.screenshot({ path: 'reader-screenshot.png', fullPage: false });
  console.log('screenshot saved -> web/reader-screenshot.png');

  // --- Lookup verification ---
  // Programmatically select a 4-char span from the first paragraph and dispatch mouseup.
  await page.evaluate(() => {
    const p = document.querySelector('.reader-body [data-pid]');
    if (!p) return;
    // Find the first text node inside.
    const tw = document.createTreeWalker(p, NodeFilter.SHOW_TEXT);
    const node = tw.nextNode();
    if (!node) return;
    const range = document.createRange();
    range.setStart(node, 0);
    range.setEnd(node, Math.min(3, node.textContent.length));
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
    p.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
  });
  // Wait for lookup index to load and render.
  await page.waitForSelector('.lookup-summary, .lookup-empty p', { timeout: 30000 });
  const summary = await page.locator('.lookup-summary, .lookup-empty').first().textContent();
  console.log('lookup summary:', summary?.trim().slice(0, 120));
  const hitCount = await page.locator('.lookup-hit').count();
  console.log('lookup hits rendered:', hitCount);

  await page.screenshot({ path: 'reader-with-lookup.png', fullPage: false });
  console.log('lookup screenshot -> web/reader-with-lookup.png');

  const ok = paragraphs > 0 && huMarkers > 0 && tradHits.length === 0 && simpHits.length > 0;
  await browser.close();
  process.exit(ok ? 0 : 1);
})().catch(e => { console.error(e); process.exit(2); });
