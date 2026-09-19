import { classifyHardGood } from "./categories";
import {
  prettySupplier,
  sourceFromChina,
  type ChinaSourceResult,
  type SourcedListing,
} from "./china-source";
import { generateDealRoomBlueprint } from "./generate-deal-room";
import { buildLandedQuote } from "./landed-cost";
import type {
  Deal,
  DealStatus,
  HardGoodsCategory,
  InquiryInput,
  SourceAttemptView,
  TimelineEvent,
} from "./types";

function id(prefix: string): string {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`;
}

function titleFromInput(
  input: InquiryInput,
  category: HardGoodsCategory,
): string {
  if (input.title?.trim()) return input.title.trim();
  const first = input.description.trim().split(/[.!\n]/)[0]?.trim();
  if (first && first.length < 80) return first;
  switch (category) {
    case "sofa":
      return "Custom upholstered sofa";
    case "lighting":
      return "Custom lighting fixture";
    case "table":
      return "Custom table";
    default:
      return "Hard goods inquiry";
  }
}

function fallbackProvider(category: HardGoodsCategory): {
  name: string;
  city: string;
} {
  switch (category) {
    case "sofa":
      return { name: "Foshan upholstery cluster (pending live match)", city: "Foshan" };
    case "lighting":
      return { name: "Zhongshan lighting corridor (pending live match)", city: "Zhongshan" };
    case "table":
      return { name: "Dongguan woodworks (pending live match)", city: "Dongguan" };
    default:
      return { name: "China hard-goods export (pending live match)", city: "Shenzhen" };
  }
}

function cityFromListing(listing?: SourcedListing): string {
  if (!listing) return "China";
  if (listing.source === "made-in-china") {
    // many Foshan furniture exporters on MIC; keep generic unless hostname hints
    const host = listing.supplierUrl || listing.url;
    if (/foshan|fs/i.test(host)) return "Foshan";
    if (/dongguan|dg/i.test(host)) return "Dongguan";
    if (/shenzhen|sz/i.test(host)) return "Shenzhen";
    if (/zhongshan|zs/i.test(host)) return "Zhongshan";
    return "China (Made-in-China)";
  }
  if (listing.source === "aliexpress") return "China (AliExpress)";
  return "China";
}

function economics(
  category: HardGoodsCategory,
  quantity: number,
  livePrice?: number,
) {
  const base = (() => {
    switch (category) {
      case "sofa":
        return {
          factory: 220 * quantity,
          cbm: 1.85 * quantity,
          lead: 18,
          moq: 1,
          dims: { width: 220, depth: 98, height: 82 },
          confidence: 0.86,
        };
      case "lighting":
        return {
          factory: 48 * quantity,
          cbm: 0.12 * quantity,
          lead: 14,
          moq: 2,
          dims: { width: 45, depth: 45, height: 28 },
          confidence: 0.81,
        };
      case "table":
        return {
          factory: 310 * quantity,
          cbm: 0.95 * quantity,
          lead: 21,
          moq: 1,
          dims: { width: 200, depth: 95, height: 75 },
          confidence: 0.84,
        };
      default:
        return {
          factory: 120 * quantity,
          cbm: 0.4 * quantity,
          lead: 16,
          moq: 1,
          dims: { width: 60, depth: 40, height: 50 },
          confidence: 0.72,
        };
    }
  })();

  if (livePrice && livePrice > 0) {
    // AliExpress is often retail-export; treat ~45% as factory-ish working number for quote draft
    const estimatedFactory = Math.max(
      40,
      Math.round((livePrice * 0.45) * quantity),
    );
    return {
      ...base,
      factory: estimatedFactory,
      confidence: Math.min(0.92, base.confidence + 0.05),
    };
  }
  return base;
}

function attemptsView(result: ChinaSourceResult): SourceAttemptView[] {
  return result.attempts.map((a) => ({
    source: a.source,
    ok: a.ok,
    status: a.status,
    detail: a.detail,
    query: a.query,
    searchedAt: a.searchedAt,
    listingCount: a.listings.length,
  }));
}

function buildLiveTimeline(
  result: ChinaSourceResult,
  top: SourcedListing | undefined,
): TimelineEvent[] {
  const now = Date.now();
  const at = (minsAgo: number) =>
    new Date(now - minsAgo * 60_000).toISOString();
  const events: TimelineEvent[] = [
    {
      id: id("evt"),
      at: at(8),
      actor: "system",
      kind: "sourced",
      title: "Live China web sourcing started",
      body: `Agent queried Chinese marketplaces for “${result.query}”.`,
      facts: { query: result.query, live: String(result.live) },
    },
  ];

  for (const [i, attempt] of result.attempts.entries()) {
    events.push({
      id: id("evt"),
      at: at(7 - i),
      actor: "agent",
      kind: attempt.ok ? "answered" : "risk",
      title: `${attempt.source}: ${attempt.status}`,
      body: attempt.detail,
      facts: {
        listings: String(attempt.listings.length),
        status: attempt.status,
      },
    });
  }

  if (top) {
    events.push({
      id: id("evt"),
      at: at(1),
      actor: "system",
      kind: "sourced",
      title: "Top live listing selected for deal room",
      body: `${top.title} — ${prettySupplier(top)}`,
      facts: {
        source: top.source,
        url: top.url,
        ...(top.priceUsd ? { price_usd: `$${top.priceUsd}` } : {}),
      },
    });
    events.push({
      id: id("evt"),
      at: at(0),
      actor: "agent",
      kind: "status",
      title: "Factory chat not opened yet",
      body: "Next step (not in this build): agent opens WeChat/WhatsApp to the supplier storefront contact. Deal room will retranscribe — client still has no direct chat.",
    });
  } else {
    events.push({
      id: id("evt"),
      at: at(0),
      actor: "system",
      kind: "risk",
      title: "No live listings returned",
      body: "Alibaba/1688 are often CAPTCHA-blocked from cloud IPs. Made-in-China / AliExpress returned empty for this query — retry or refine the brief.",
    });
  }

  return events;
}

export async function createDealFromInquiry(
  input: InquiryInput,
  options?: { skipLiveSource?: boolean },
): Promise<Deal> {
  const blob = `${input.title ?? ""} ${input.description} ${input.sourceUrl ?? ""}`;
  const category = classifyHardGood(blob);
  const title = titleFromInput(input, category);

  let sourced: ChinaSourceResult | undefined;
  if (!options?.skipLiveSource) {
    try {
      sourced = await sourceFromChina({
        description: `${title}. ${input.description}`,
        category,
      });
    } catch (e) {
      sourced = {
        query: title,
        attempts: [
          {
            source: "alibaba",
            ok: false,
            status: "error",
            detail: e instanceof Error ? e.message : "Sourcing failed",
            listings: [],
            searchedAt: new Date().toISOString(),
            query: title,
          },
        ],
        best: [],
        live: false,
      };
    }
  }

  const top = sourced?.best[0];
  const livePrice = sourced?.best.find((l) => l.priceUsd)?.priceUsd;
  const eco = economics(category, input.quantity, livePrice);
  const provider = top
    ? { name: prettySupplier(top), city: cityFromListing(top) }
    : fallbackProvider(category);

  const quote = buildLandedQuote({
    factoryPriceUsd: eco.factory,
    cbm: eco.cbm,
    destinationCountry: input.destinationCountry,
    leadTimeDays: eco.lead,
    moq: eco.moq,
  });

  const blueprint = generateDealRoomBlueprint({
    title,
    description: input.description,
    category,
    quantity: input.quantity,
    destinationCountry: input.destinationCountry,
    providerCity: provider.city,
    confidence: sourced?.live ? eco.confidence : Math.max(0.4, eco.confidence - 0.2),
    dims: eco.dims,
    cbm: eco.cbm,
    includeLiveListings: Boolean(sourced?.best.length),
  });

  const now = new Date().toISOString();
  const timeline = sourced
    ? buildLiveTimeline(sourced, top)
    : [
        {
          id: id("evt"),
          at: now,
          actor: "system" as const,
          kind: "status" as const,
          title: "Live sourcing skipped",
          body: "Deal created without marketplace scrape.",
        },
      ];

  return {
    id: id("deal"),
    createdAt: now,
    updatedAt: now,
    status: sourced?.live ? "quoted" : "sourcing",
    category,
    title,
    description: input.description,
    imageDataUrl: input.imageDataUrl ?? top?.imageUrl,
    sourceUrl: input.sourceUrl,
    quantity: input.quantity,
    destinationCountry: input.destinationCountry,
    budgetUsd: input.budgetUsd,
    providerName: provider.name,
    providerCity: provider.city,
    sourcing: sourced
      ? {
          live: sourced.live,
          query: sourced.query,
          attempts: attemptsView(sourced),
          listings: sourced.best,
        }
      : undefined,
    blueprint,
    timeline,
    quote,
    specLock: [
      { label: "Product", value: title },
      {
        label: "Origin signal",
        value: top
          ? `${top.source}: ${prettySupplier(top)}`
          : "No live listing yet",
      },
      {
        label: "Listing URL",
        value: top?.url ?? "—",
      },
      { label: "Sampling", value: "None — hard goods photo/spec lock only" },
      {
        label: "Client channel",
        value: "Deal room only (no factory chat)",
      },
    ],
  };
}

export function applyDealAction(
  deal: Deal,
  action: "approve" | "request_change" | "reject",
  note?: string,
): Deal {
  const now = new Date().toISOString();
  const event = (
    partial: Omit<TimelineEvent, "id" | "at">,
  ): TimelineEvent => ({
    id: id("evt"),
    at: now,
    ...partial,
  });

  if (action === "approve") {
    return {
      ...deal,
      status: "deposit_due" as DealStatus,
      updatedAt: now,
      timeline: [
        ...deal.timeline,
        event({
          actor: "client",
          kind: "agreed",
          title: "Client approved landed quote",
          body: "Deposit gate opened. Factory will not see client identity — agent continues from the live listing contact path.",
        }),
      ],
    };
  }

  if (action === "reject") {
    return {
      ...deal,
      status: "rejected",
      updatedAt: now,
      timeline: [
        ...deal.timeline,
        event({
          actor: "client",
          kind: "status",
          title: "Client rejected this match",
          body: note?.trim() || "Deal closed without outreach continuation.",
        }),
      ],
    };
  }

  const newFactory = Math.max(
    40,
    Math.round(deal.quote.factoryPriceUsd * 0.94),
  );
  const freightModule = deal.blueprint.modules.find(
    (m) => m.type === "freight_volume",
  );
  const cbm =
    freightModule && freightModule.type === "freight_volume"
      ? freightModule.cbm
      : 0.5;
  const quote = buildLandedQuote({
    factoryPriceUsd: newFactory,
    cbm,
    destinationCountry: deal.destinationCountry,
    leadTimeDays: deal.quote.leadTimeDays,
    moq: deal.quote.moq,
    version: deal.quote.version + 1,
  });

  return {
    ...deal,
    status: "negotiating",
    updatedAt: now,
    quote,
    timeline: [
      ...deal.timeline,
      event({
        actor: "client",
        kind: "asked",
        title: "Client requested a change",
        body:
          note?.trim() ||
          "Asked agent to push price / revise a commercial term.",
      }),
      event({
        actor: "agent",
        kind: "pushed",
        title: "Agent queued supplier follow-up",
        body: deal.sourcing?.listings[0]
          ? `Would message contact on ${deal.sourcing.listings[0].source}: ${deal.sourcing.listings[0].url}`
          : "No live listing URL on file — refine sourcing first.",
      }),
      event({
        actor: "factory",
        kind: "agreed",
        title: "Draft revised FOB (simulated pending chat bridge)",
        body: `Working FOB $${newFactory}. WhatsApp/WeChat bridge not connected yet — quote refreshed to v${quote.version} for deal-room UX.`,
        facts: {
          fob_usd: `$${newFactory}`,
          quote_version: String(quote.version),
        },
      }),
    ],
  };
}
