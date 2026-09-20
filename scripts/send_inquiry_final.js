const { chromium } = require("playwright");
const fs = require("fs");

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({
    userAgent:
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
  });

  const email = "paul.faverjon@student-cs.fr";
  const message = [
    "Hello Ms. He,",
    "",
    "We are China Access, a sourcing agent helping a US client source hard goods.",
    "Request: FOB quote for one cloud-style 3-seat fabric sofa (~220cm), linen or performance fabric, export packed to USA.",
    "Please reply with: FOB USD, MOQ for 1 unit, lead time (days), fabric options, and CBM.",
    "",
    "Thank you,",
    "China Access agent",
  ].join("\n");

  const result = { ok: false, steps: [], evidence: {} };

  await page.goto(
    "https://yefeifurniture.en.made-in-china.com/contact-info.html",
    { waitUntil: "domcontentloaded", timeout: 60000 },
  );
  await page.waitForTimeout(2000);
  await page.fill("#J-quick-inquiry-input", email);
  await page.fill("#inquiryContent", message);
  await page.click("#inquirySend");

  const frameHandle = await page.waitForSelector(
    'iframe[src*="showQuickForm"]',
    { timeout: 20000 },
  );
  const frame = await frameHandle.contentFrame();
  if (!frame) throw new Error("no iframe");
  await frame.waitForSelector("#senderMail", { timeout: 15000 });
  result.steps.push("iframe open");

  await frame.fill("#senderMail", email);
  await frame.locator('input[name="senderName"]').fill("Paul Faverjon");
  await frame.locator('input[name="senderCom"]').fill("China Access");
  await frame.fill("#senderMobile", "5550100123");
  result.steps.push("identity filled");

  await page.screenshot({
    path: "/workspace/scripts/outreach_ready.png",
    fullPage: true,
  });

  const responses = [];
  page.on("response", async (r) => {
    if (/sendInquiry|inquiry|quick|message|showQuickForm/i.test(r.url())) {
      let body = "";
      try {
        body = (await r.text()).slice(0, 800);
      } catch {}
      responses.push({ status: r.status(), url: r.url(), body });
    }
  });

  await Promise.all([
    frame.waitForNavigation({ timeout: 20000 }).catch(() => null),
    frame.click("#submitSend"),
  ]);
  result.steps.push("submitted");
  await page.waitForTimeout(5000);

  const bodies = [];
  for (const f of page.frames()) {
    const text = await f
      .locator("body")
      .innerText()
      .catch(() => "");
    if (text && /thank|success|sent|inquiry|trust|error|fail|verify/i.test(text)) {
      bodies.push({ url: f.url(), text: text.slice(0, 2000) });
    }
  }
  for (const p of page.context().pages()) {
    bodies.push({
      url: p.url(),
      title: await p.title(),
      text: (await p.locator("body").innerText().catch(() => "")).slice(0, 2000),
    });
  }

  result.evidence.responses = responses;
  result.evidence.bodies = bodies;
  result.ok = bodies.some((b) =>
    /thank you|successfully|has been sent|inquiry has been|sent successfully|we have received/i.test(
      b.text || "",
    ),
  );
  // Also treat disappearance of trust modal + no error as soft success if response 200
  if (
    !result.ok &&
    responses.some((r) => r.status >= 200 && r.status < 400 && /send/i.test(r.url))
  ) {
    result.softOk = true;
  }

  await page.screenshot({
    path: "/workspace/scripts/outreach_submitted.png",
    fullPage: true,
  });
  fs.writeFileSync(
    "/workspace/scripts/outreach_final.json",
    JSON.stringify(result, null, 2),
  );
  console.log(JSON.stringify(result, null, 2));
  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
