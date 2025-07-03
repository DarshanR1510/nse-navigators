// playwright_mcp_server.js
const { chromium } = require('playwright');

async function main() {
  const url = process.argv[2];
  if (!url) {
    console.error('No URL provided');
    process.exit(1);
  }
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });
  // Wait for main content to load (customize selector if needed)
  await page.waitForTimeout(2000);
  const html = await page.content();
  console.log(html);
  await browser.close();
}
main().catch(e => {
  console.error(e);
  process.exit(1);
});
