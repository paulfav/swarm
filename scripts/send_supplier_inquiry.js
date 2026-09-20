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

  const result = {
    ok: false,
    steps: [],
    finalUrl: "",
    finalTitle: "",
    bodySnippet: "",
  };

  await page.goto(
    "https://yefeifurniture.en.made-in-china.com/contact-info.html",
    { waitUntil: "domcontentloaded", timeout: 60000 },
  );
  await page.waitForTimeout(2000);
  result.steps.push("opened contact page");

  await page.fill("#J-quick-inquiry-input", email);
  await page.fill("#inquiryContent", message);
  result.steps.push("filled email + message");

  await page.click("#inquirySend");
  await page.waitForTimeout(2500);
  result.steps.push("clicked Send, waiting for trust modal");

  // Wait for trust modal
  await page.waitForSelector("text=build trust", { timeout: 15000 });
  await page.screenshot({
    path: "/workspace/scripts/outreach_modal.png",
    fullPage: true,
  });

  // Fill modal fields — they may not have stable IDs; locate by placeholder/label
  const filled = await page.evaluate(() => {
    const visible = (el) => {
      const r = el.getBoundingClientRect();
      return r.width > 0 && r.height > 0;
    };
    const inputs = [...document.querySelectorAll("input, select, textarea")].filter(
      visible,
    );
    const out = [];
    for (const el of inputs) {
      const key = `${el.name}|${el.id}|${el.placeholder}|${el.type}`.toLowerCase();
      out.push({
        key,
        name: el.name,
        id: el.id,
        ph: el.placeholder,
        type: el.type,
        tag: el.tagName,
      });
    }
    return out;
  });
  console.log("modal/visible fields", JSON.stringify(filled, null, 2));
  result.steps.push(`found ${filled.length} visible fields`);

  // Full name
  const nameInput = page
    .locator(
      'input[placeholder*="Name" i], input[name*="Name" i], input[name="senderName"], #senderName',
    )
    .first();
  if (await nameInput.count()) {
    await nameInput.fill("Paul Faverjon");
    result.steps.push("filled name");
  }

  // Company
  const companyInput = page
    .locator(
      'input[placeholder*="Company" i], input[name*="comName" i], input[name="comName"], #comName',
    )
    .first();
  if (await companyInput.count()) {
    await companyInput.fill("China Access");
    result.steps.push("filled company");
  }

  // Phone / mobile
  const phoneInput = page
    .locator(
      'input[placeholder*="phone" i], input[placeholder*="Mobile" i], input[name*="Mobile" i], input[name="senderMobile"], #senderMobile',
    )
    .first();
  if (await phoneInput.count()) {
    await phoneInput.fill("5550100123");
    result.steps.push("filled phone");
  }

  // Ensure email in modal if separate
  const modalEmail = page
    .locator(
      '#senderMailDialog, input[name="senderMailDialog"], .dialog input[type="text"][name*="ail" i]',
    )
    .first();
  if (await modalEmail.count()) {
    await modalEmail.fill(email);
    result.steps.push("filled modal email");
  }

  await page.screenshot({
    path: "/workspace/scripts/outreach_modal_filled.png",
    fullPage: true,
  });

  // Click red Send inside the modal (not the disabled background one)
  const modalSend = page
    .locator(".dialog button, .dlg button, [class*='dialog'] button, .ok-btn")
    .filter({ hasText: /^Send$/i })
    .first();

  const responses = [];
  page.on("response", async (r) => {
    if (/sendInquiry|inquiry|message|quickpost/i.test(r.url())) {
      responses.push({ status: r.status(), url: r.url() });
    }
  });

  if (await modalSend.count()) {
    await modalSend.click();
    result.steps.push("clicked modal Send");
  } else {
    // fallback: click any visible Send that is not disabled
    await page.evaluate(() => {
      const buttons = [...document.querySelectorAll("button, input[type='button'], input[type='submit']")];
      const btn = buttons.find((b) => {
        const r = b.getBoundingClientRect();
        const t = (b.innerText || b.value || "").trim();
        return r.width > 0 && /^send$/i.test(t) && !b.className.toString().includes("disabled");
      });
      if (btn) btn.click();
    });
    result.steps.push("clicked fallback Send");
  }

  await page.waitForTimeout(6000);

  // Check for new pages
  const pages = page.context().pages();
  const top = pages[pages.length - 1];
  result.finalUrl = top.url();
  result.finalTitle = await top.title();
  result.bodySnippet = (await top.locator("body").innerText()).slice(0, 2000);
  result.responses = responses;

  const success = /thank|success|sent|submitted|received your/i.test(
    result.bodySnippet + " " + result.finalTitle,
  );
  result.ok = success;
  result.steps.push(success ? "SUCCESS detected" : "no clear success text");

  await top.screenshot({
    path: "/workspace/scripts/outreach_result.png",
    fullPage: true,
  });
  fs.writeFileSync(
    "/workspace/scripts/outreach_result.json",
    JSON.stringify(result, null, 2),
  );
  console.log(JSON.stringify(result, null, 2));
  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
