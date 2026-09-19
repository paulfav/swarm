import { chromium, type Browser, type Page } from "playwright";
import type { HardGoodsCategory } from "./types";

export interface SourcedListing {
  source: "made-in-china" | "aliexpress" | "alibaba";
  title: string;
  url: string;
  priceUsd?: number;
  imageUrl?: string;
  supplierName?: string;
  supplierUrl?: string;
  rawPriceText?: string;
}

export interface SourceAttempt {
  source: SourcedListing["source"];
  ok: boolean;
  status: "ok" | "captcha" | "empty" | "error";
  detail: string;
  listings: SourcedListing[];
  searchedAt: string;
  query: string;
}

export interface ChinaSourceResult {
  query: string;
  attempts: SourceAttempt[];
  best: SourcedListing[];
  live: boolean;
}

function idFromMicHost(host: string): string {
  // fscooc.en.made-in-china.com → fscooc
  const m = host.match(/^([a-z0-9-]+)\.en\.made-in-china\.com/i);
  return m?.[1] ?? host;
}

function parsePrice(text: string): number | undefined {
  const cleaned = text.replace(/,/g, "");
  const m = cleaned.match(/\$?\s*(\d+(?:\.\d+)?)/);
  if (!m) return undefined;
  const n = Number(m[1]);
  return Number.isFinite(n) ? n : undefined;
}

function buildQuery(
  description: string,
  category: HardGoodsCategory,
): string {
  const words = description
    .replace(/https?:\/\/\S+/g, "")
    .replace(/[^a-zA-Z0-9\s-]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2)
    .slice(0, 8);
  const base = words.join(" ").trim();
  if (base) return base;
  switch (category) {
    case "sofa":
      return "linen sofa living room";
    case "lighting":
      return "brass pendant lamp";
    case "table":
      return "oak dining table";
    default:
      return "furniture wholesale china";
  }
}

async function withBrowser<T>(fn: (page: Page) => Promise<T>): Promise<T> {
  let browser: Browser | null = null;
  try {
    browser = await chromium.launch({ headless: true });
    const page = await browser.newPage({
      userAgent:
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
      locale: "en-US",
    });
    page.setDefaultTimeout(45000);
    return await fn(page);
  } finally {
    await browser?.close().catch(() => undefined);
  }
}

async function searchMadeInChina(
  page: Page,
  query: string,
): Promise<SourceAttempt> {
  const searchedAt = new Date().toISOString();
  const slug = query.trim().replace(/\s+/g, "_");
  const url = `https://www.made-in-china.com/products-search/hot-china-products/${encodeURIComponent(slug)}.html`;
  try {
    const res = await page.goto(url, { waitUntil: "domcontentloaded" });
    await page.waitForTimeout(3500);
    const title = await page.title();
    if (/captcha|verify/i.test(title)) {
      return {
        source: "made-in-china",
        ok: false,
        status: "captcha",
        detail: `CAPTCHA on Made-in-China (${title})`,
        listings: [],
        searchedAt,
        query,
      };
    }

    const listings = await page.evaluate(() => {
      const out: {
        href: string;
        text: string;
        supplierName?: string;
        supplierUrl?: string;
        imageUrl?: string;
      }[] = [];
      const seen = new Set<string>();
      for (const a of Array.from(document.querySelectorAll("a"))) {
        const href = (a as HTMLAnchorElement).href || "";
        const text = ((a as HTMLAnchorElement).innerText || "")
          .trim()
          .replace(/\s+/g, " ");
        if (
          !/\/product\//i.test(href) ||
          text.length < 18 ||
          text.length > 200 ||
          seen.has(href.split("?")[0]!)
        ) {
          continue;
        }
        seen.add(href.split("?")[0]!);
        let supplierName: string | undefined;
        let supplierUrl: string | undefined;
        try {
          const u = new URL(href);
          const m = u.hostname.match(/^([a-z0-9-]+)\.en\.made-in-china\.com/i);
          if (m) {
            supplierName = m[1];
            supplierUrl = `${u.protocol}//${u.hostname}/`;
          }
        } catch {
          /* ignore */
        }
        const root =
          a.closest("li, article, .prod-item, .list-node, .sr-proItem") ||
          a.parentElement;
        const img = root?.querySelector("img") as HTMLImageElement | null;
        out.push({
          href: href.split("?")[0]!,
          text,
          supplierName,
          supplierUrl,
          imageUrl: img?.src || undefined,
        });
        if (out.length >= 8) break;
      }
      return out;
    });

    const mapped: SourcedListing[] = listings.map((l) => ({
      source: "made-in-china" as const,
      title: l.text,
      url: l.href,
      supplierName: l.supplierName
        ? `${l.supplierName} (Made-in-China)`
        : undefined,
      supplierUrl: l.supplierUrl,
      imageUrl: l.imageUrl,
    }));

    return {
      source: "made-in-china",
      ok: mapped.length > 0,
      status: mapped.length ? "ok" : "empty",
      detail: mapped.length
        ? `Found ${mapped.length} factory listings`
        : `No product cards parsed (HTTP ${res?.status() ?? "?"})`,
      listings: mapped,
      searchedAt,
      query,
    };
  } catch (e) {
    return {
      source: "made-in-china",
      ok: false,
      status: "error",
      detail: e instanceof Error ? e.message : "Made-in-China scrape failed",
      listings: [],
      searchedAt,
      query,
    };
  }
}

async function searchAliExpress(
  page: Page,
  query: string,
): Promise<SourceAttempt> {
  const searchedAt = new Date().toISOString();
  const url = `https://www.aliexpress.com/w/wholesale-${encodeURIComponent(
    query.trim().replace(/\s+/g, "-"),
  )}.html`;
  try {
    const res = await page.goto(url, { waitUntil: "domcontentloaded" });
    await page.waitForTimeout(4000);
    const title = await page.title();
    if (/captcha|verify/i.test(title)) {
      return {
        source: "aliexpress",
        ok: false,
        status: "captcha",
        detail: "CAPTCHA on AliExpress search",
        listings: [],
        searchedAt,
        query,
      };
    }

    const listings = await page.evaluate(() => {
      const cards: {
        href: string;
        title: string;
        price: string;
        image: string;
      }[] = [];
      const seen = new Set<string>();
      for (const a of Array.from(
        document.querySelectorAll('a[href*="/item/"]'),
      )) {
        const href = ((a as HTMLAnchorElement).href || "").split("?")[0]!;
        if (seen.has(href)) continue;
        const text = ((a as HTMLAnchorElement).innerText || "")
          .trim()
          .replace(/\s+/g, " ");
        if (text.length < 15) continue;
        seen.add(href);
        const root =
          a.closest(
            '[class*="card"], [class*="product"], [class*="item"], li, article',
          ) || a.parentElement;
        const priceEl = root?.querySelector(
          '[class*="price"], [class*="Price"]',
        );
        const img = root?.querySelector("img") as HTMLImageElement | null;
        cards.push({
          href,
          title: text.slice(0, 160),
          price: (priceEl?.textContent || "").trim().replace(/\s+/g, " "),
          image: img?.src || "",
        });
        if (cards.length >= 8) break;
      }
      return cards;
    });

    const mapped: SourcedListing[] = listings.map((l) => ({
      source: "aliexpress" as const,
      title: l.title.replace(/\$[\d.,]+.*$/, "").trim() || l.title,
      url: l.href,
      rawPriceText: l.price || undefined,
      priceUsd: l.price ? parsePrice(l.price) : undefined,
      imageUrl: l.image || undefined,
      supplierName: "AliExpress seller (CN export retail)",
    }));

    return {
      source: "aliexpress",
      ok: mapped.length > 0,
      status: mapped.length ? "ok" : "empty",
      detail: mapped.length
        ? `Found ${mapped.length} AliExpress listings`
        : `No item cards parsed (HTTP ${res?.status() ?? "?"})`,
      listings: mapped,
      searchedAt,
      query,
    };
  } catch (e) {
    return {
      source: "aliexpress",
      ok: false,
      status: "error",
      detail: e instanceof Error ? e.message : "AliExpress scrape failed",
      listings: [],
      searchedAt,
      query,
    };
  }
}

async function searchAlibaba(
  page: Page,
  query: string,
): Promise<SourceAttempt> {
  const searchedAt = new Date().toISOString();
  const url = `https://www.alibaba.com/trade/search?fsb=y&IndexArea=product_en&SearchText=${encodeURIComponent(query)}&tab=all`;
  try {
    await page.goto(url, { waitUntil: "domcontentloaded" });
    await page.waitForTimeout(3000);
    const title = await page.title();
    if (/captcha|verify/i.test(title)) {
      return {
        source: "alibaba",
        ok: false,
        status: "captcha",
        detail:
          "Alibaba blocked this datacenter IP with CAPTCHA — needs residential proxy or official API keys",
        listings: [],
        searchedAt,
        query,
      };
    }
    const count = await page
      .locator('a[href*="product-detail"]')
      .count()
      .catch(() => 0);
    return {
      source: "alibaba",
      ok: count > 0,
      status: count ? "ok" : "empty",
      detail: count
        ? `Found ${count} product-detail links`
        : "No Alibaba product links parsed",
      listings: [],
      searchedAt,
      query,
    };
  } catch (e) {
    return {
      source: "alibaba",
      ok: false,
      status: "error",
      detail: e instanceof Error ? e.message : "Alibaba scrape failed",
      listings: [],
      searchedAt,
      query,
    };
  }
}

/**
 * Live China marketplace sourcing.
 * Primary: Made-in-China (factory storefronts) + AliExpress (price signal).
 * Alibaba/1688 are attempted but often CAPTCHA-blocked from cloud IPs.
 */
export async function sourceFromChina(input: {
  description: string;
  category: HardGoodsCategory;
}): Promise<ChinaSourceResult> {
  const query = buildQuery(input.description, input.category);

  return withBrowser(async (page) => {
    const attempts: SourceAttempt[] = [];
    attempts.push(await searchMadeInChina(page, query));
    attempts.push(await searchAliExpress(page, query));
    attempts.push(await searchAlibaba(page, query));

    const best: SourcedListing[] = [];
    const seen = new Set<string>();
    // Prefer Made-in-China (closer to factories), then AliExpress
    for (const source of ["made-in-china", "aliexpress", "alibaba"] as const) {
      const attempt = attempts.find((a) => a.source === source);
      for (const listing of attempt?.listings ?? []) {
        if (seen.has(listing.url)) continue;
        seen.add(listing.url);
        best.push(listing);
        if (best.length >= 6) break;
      }
      if (best.length >= 6) break;
    }

    return {
      query,
      attempts,
      best,
      live: best.length > 0,
    };
  });
}

export function prettySupplier(listing: SourcedListing): string {
  if (listing.supplierName) return listing.supplierName;
  try {
    if (listing.source === "made-in-china") {
      return `${idFromMicHost(new URL(listing.url).hostname)} (Made-in-China)`;
    }
  } catch {
    /* ignore */
  }
  return listing.source;
}
