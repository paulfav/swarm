const { chromium } = require("playwright");

(async () => {
  console.log("launching");
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({
    userAgent:
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
  });
  const email = "paul.faverjon@student-cs.fr";
  const message =
    "Hello Ms. He, we are China Access sourcing agent for a US client. Please quote FOB for one cloud-style 3-seat fabric sofa ~220cm, MOQ 1, lead time, fabric options, CBM for USA export. Thank you - China Access agent.";

  await page.goto(
    "https://yefeifurniture.en.made-in-china.com/contact-info.html",
    { waitUntil: "domcontentloaded", timeout: 60000 },
  );
  await page.waitForTimeout(2000);
  await page.fill("#J-quick-inquiry-input", email);
  await page.fill("#inquiryContent", message);

  let second = "";
  page.on("response", async (r) => {
    if (/qkSendStep=second/i.test(r.url())) {
      second = await r.text().catch(() => "");
      console.log("got second step", r.status(), second.slice(0, 300));
    }
  });

  await page.click("#inquirySend");
  await page.waitForTimeout(5000);
  await page.screenshot({ path: "/tmp/mic_trust_modal.png", fullPage: true });

  const text = await page.locator("body").innerText();
  console.log("has trust modal", /build trust/i.test(text));
  console.log(text.slice(0, 1500));

  const controls = await page.evaluate(() =>
    [...document.querySelectorAll("input,textarea,button,a")]
      .filter((el) => el.getBoundingClientRect().width > 0)
      .map((el) => ({
        id: el.id,
        name: el.name,
        ph: el.placeholder,
        type: el.type,
        href: el.href || "",
        text: (el.innerText || "").slice(0, 40),
        cls: (el.className || "").toString().slice(0, 50),
      })),
  );
  console.log(JSON.stringify(controls, null, 2));
  console.log("SECOND RAW", second.slice(0, 3000));

  // Try to interact with dialog ok button / fill senderMailDialog
  if (await page.locator("#senderMailDialog").count()) {
    await page.fill("#senderMailDialog", email);
    console.log("filled senderMailDialog");
  }
  if (await page.locator(".ok-btn, button.ok-btn, .btn:has-text('OK')").count()) {
    await page.click(".ok-btn, button.ok-btn").catch(() => {});
    console.log("clicked ok");
    await page.waitForTimeout(4000);
    console.log("after ok", page.url(), await page.title());
    console.log((await page.locator("body").innerText()).slice(0, 1200));
    await page.screenshot({ path: "/tmp/mic_after_ok.png", fullPage: true });
  }

  // Also try direct navigation to second-step URL pattern from network
  const m = second.match(/https?:\/\/[^"'\\\s]+sendInquiry[^"'\\\s]+/);
  if (m) console.log("found url in second", m[0]);

  await browser.close();
  console.log("done");
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
