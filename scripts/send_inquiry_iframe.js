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
  result.steps.push("filled contact form");

  await page.click("#inquirySend");
  result.steps.push("clicked Send");

  // Wait for quick-form iframe
  const frameHandle = await page.waitForSelector(
    'iframe[src*="showQuickForm"]',
    { timeout: 20000 },
  );
  const frame = await frameHandle.contentFrame();
  if (!frame) throw new Error("no iframe content");
  result.steps.push("quick-form iframe ready");

  await frame.waitForTimeout(1500);
  const fields = await frame.$$eval("input,textarea,select,button", (els) =>
    els.map((el) => ({
      name: el.name,
      id: el.id,
      ph: el.placeholder,
      type: el.type,
      text: (el.innerText || "").slice(0, 40),
      value: (el.value || "").slice(0, 60),
    })),
  );
  console.log("iframe fields", JSON.stringify(fields, null, 2));
  result.evidence.iframeFields = fields;

  // Fill identity
  const fillIf = async (sel, value, label) => {
    const loc = frame.locator(sel).first();
    if ((await loc.count()) > 0) {
      await loc.fill(value);
      result.steps.push(`filled ${label}`);
      return true;
    }
    return false;
  };

  await fillIf(
    '#senderMail, input[name="senderMail"], input[type="email"]',
    email,
    "email",
  );
  await fillIf(
    '#senderName, input[name="senderName"], input[placeholder*="Name" i]',
    "Paul Faverjon",
    "name",
  );
  await fillIf(
    '#comName, input[name="comName"], input[placeholder*="Company" i]',
    "China Access",
    "company",
  );
  await fillIf(
    '#senderMobile, input[name="senderMobile"], input[placeholder*="phone" i], input[placeholder*="Mobile" i]',
    "5550100123",
    "phone",
  );

  // Country if select
  const country = frame.locator(
    'select[name*="country" i], select[id*="country" i]',
  );
  if (await country.count()) {
    await country.selectOption({ label: /United States/i }).catch(async () => {
      await country.selectOption({ value: "US" }).catch(() => {});
    });
    result.steps.push("set country");
  }

  await page.screenshot({
    path: "/workspace/scripts/outreach_iframe_filled.png",
    fullPage: true,
  });

  const responses = [];
  page.on("response", async (r) => {
    if (/sendInquiry|inquiry|quick|message/i.test(r.url())) {
      let body = "";
      try {
        body = (await r.text()).slice(0, 500);
      } catch {}
      responses.push({ status: r.status(), url: r.url(), body });
    }
  });

  // Click Send in iframe
  const sendBtn = frame
    .locator("button, input[type='submit'], input[type='button']")
    .filter({ hasText: /send/i })
    .first();
  if (await sendBtn.count()) {
    await sendBtn.click();
    result.steps.push("clicked iframe Send");
  } else {
    await frame.evaluate(() => {
      const b = [...document.querySelectorAll("button, input")].find((el) =>
        /send/i.test(el.innerText || el.value || ""),
      );
      if (b) b.click();
    });
    result.steps.push("clicked iframe Send via evaluate");
  }

  await page.waitForTimeout(8000);
  result.evidence.responses = responses;

  // Check all pages / iframe after submit
  const bodies = [];
  for (const p of page.context().pages()) {
    bodies.push({
      url: p.url(),
      title: await p.title(),
      text: (await p.locator("body").innerText().catch(() => "")).slice(0, 1500),
    });
  }
  for (const f of page.frames()) {
    if (/showQuickForm|sendInquiry/i.test(f.url())) {
      bodies.push({
        url: f.url(),
        title: "frame",
        text: await f.locator("body").innerText().catch(() => ""),
      });
    }
  }
  result.evidence.bodies = bodies;
  result.ok = bodies.some((b) =>
    /thank|success|sent|submitted|received your inquiry|inquiry has been/i.test(
      b.text + " " + b.title,
    ),
  );
  result.steps.push(result.ok ? "SUCCESS" : "no success marker yet");

  await page.screenshot({
    path: "/workspace/scripts/outreach_after_iframe_send.png",
    fullPage: true,
  });
  fs.writeFileSync(
    "/workspace/scripts/outreach_iframe_result.json",
    JSON.stringify(result, null, 2),
  );
  console.log(JSON.stringify(result, null, 2));
  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
