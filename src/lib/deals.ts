import { sendWeChat } from "./channels/wechat";
import { sendWhatsApp } from "./channels/whatsapp";
import { classifyHardGood } from "./categories";
import {
  prettySupplier,
  sourceFromChina,
  type ChinaSourceResult,
  type SourcedListing,
} from "./china-source";
import { generateDealRoomBlueprint } from "./generate-deal-room";
import { buildLandedQuote } from "./landed-cost";
import { appendOutboundThread } from "./replies";
import { scrapeSupplierContacts } from "./supplier-contacts";
import {
  buildQuoteRequestMessage,
  contactMadeInChinaSupplier,
} from "./supplier-outreach";
import type {
  Deal,
  DealStatus,
  HardGoodsCategory,
  InquiryInput,
  OutreachRecord,
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
      title: "Ready to contact supplier",
      body: "Click Contact supplier in the deal room to send a real Made-in-China inquiry. Replies go to the agent email and get retranscribed here — client never chats directly.",
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
          ? `Will use Contact supplier on ${deal.sourcing.listings[0].source}: ${deal.sourcing.listings[0].url}`
          : "No live listing URL on file — refine sourcing first.",
      }),
      event({
        actor: "system",
        kind: "status",
        title: "Quote draft adjusted locally",
        body: `Working FOB $${newFactory} (v${quote.version}). Use Contact supplier to send a real Made-in-China inquiry.`,
        facts: {
          fob_usd: `$${newFactory}`,
          quote_version: String(quote.version),
        },
      }),
    ],
  };
}

/** Multi-channel supplier contact: MIC inquiry + WhatsApp/WeChat when available. */
export async function contactSupplierOnDeal(deal: Deal): Promise<Deal> {
  const listing =
    deal.sourcing?.listings.find((l) => l.source === "made-in-china") ||
    deal.sourcing?.listings[0];

  const now = new Date().toISOString();
  const event = (
    partial: Omit<TimelineEvent, "id" | "at">,
  ): TimelineEvent => ({
    id: id("evt"),
    at: now,
    ...partial,
  });

  if (!listing || listing.source !== "made-in-china") {
    return {
      ...deal,
      updatedAt: now,
      timeline: [
        ...deal.timeline,
        event({
          actor: "system",
          kind: "risk",
          title: "Cannot contact supplier yet",
          body: "Need a Made-in-China listing on this deal. Re-run sourcing or pick a MIC factory card.",
        }),
      ],
    };
  }

  const contacts =
    deal.contacts?.pageUrl === listing.url
      ? deal.contacts
      : await scrapeSupplierContacts(listing.url).catch(() => deal.contacts);

  const message = buildQuoteRequestMessage({
    productTitle: deal.title,
    description: deal.description,
    quantity: deal.quantity,
    destinationCountry: deal.destinationCountry,
    contactName: contacts?.contactPerson,
  });

  let updated: Deal = {
    ...deal,
    contacts: contacts || deal.contacts,
    updatedAt: now,
  };

  const outreach: OutreachRecord[] = [...(deal.outreach ?? [])];
  const timeline: TimelineEvent[] = [...deal.timeline];

  // 1) Made-in-China inquiry (primary, proven)
  const mic = await contactMadeInChinaSupplier({
    listingUrl: listing.url,
    supplierName: listing.supplierName,
    productTitle: listing.title,
    message,
  });
  outreach.push({
    ok: mic.ok,
    channel: "made-in-china-inquiry",
    supplierName: mic.supplierName,
    contactPerson: mic.contactPerson,
    listingUrl: listing.url,
    message,
    identityEmail: mic.identityEmail,
    successUrl: mic.successUrl,
    inquiryId: mic.inquiryId,
    steps: mic.steps,
    error: mic.error,
    sentAt: mic.sentAt,
  });

  if (mic.ok) {
    updated = appendOutboundThread(updated, {
      at: mic.sentAt,
      channel: "made-in-china-inquiry",
      body: message,
      retranscription: `Sent Made-in-China inquiry to ${mic.contactPerson || "supplier"} (${mic.supplierName || listing.supplierName || "factory"}).`,
      meta: {
        inquiry_id: mic.inquiryId || "",
        success_url: mic.successUrl || "",
      },
    });
    timeline.push(
      event({
        actor: "agent",
        kind: "asked",
        title: "Agent messaged supplier on Made-in-China",
        body: `Sent FOB quote request to ${mic.contactPerson || "supplier contact"} at ${mic.supplierName || listing.supplierName || "factory"}.`,
        facts: {
          channel: "made-in-china-inquiry",
          contact: mic.contactPerson || "—",
          inquiry_id: mic.inquiryId || "—",
          reply_to: mic.identityEmail,
        },
      }),
    );
  } else {
    timeline.push(
      event({
        actor: "agent",
        kind: "risk",
        title: "Made-in-China outreach failed",
        body: mic.error || "Inquiry did not confirm success",
        facts: { listing: listing.url },
      }),
    );
  }

  // 2) WhatsApp if number found
  if (contacts?.whatsapp) {
    const wa = await sendWhatsApp({ to: contacts.whatsapp, body: message });
    outreach.push({
      ok: wa.ok,
      channel: "whatsapp",
      listingUrl: listing.url,
      message,
      whatsappTo: wa.to,
      deepLink: wa.deepLink,
      error: wa.error,
      sentAt: wa.sentAt,
      contactPerson: contacts.contactPerson,
      supplierName: listing.supplierName,
    });
    if (wa.ok) {
      updated = appendOutboundThread(updated, {
        at: wa.sentAt,
        channel: "whatsapp",
        body: message,
        retranscription: `WhatsApp sent to +${wa.to} via Twilio.`,
        meta: { sid: wa.messageSid || "", to: wa.to },
      });
      timeline.push(
        event({
          actor: "agent",
          kind: "asked",
          title: "Agent messaged supplier on WhatsApp",
          body: `Twilio WhatsApp delivered to +${wa.to}.`,
          facts: { channel: "whatsapp", to: wa.to },
        }),
      );
    } else {
      timeline.push(
        event({
          actor: "agent",
          kind: "status",
          title: "WhatsApp ready (not auto-sent)",
          body: wa.error || "WhatsApp deep link prepared for agent.",
          facts: {
            channel: "whatsapp",
            to: wa.to,
            deep_link: wa.deepLink || "—",
          },
        }),
      );
    }
  } else {
    timeline.push(
      event({
        actor: "system",
        kind: "status",
        title: "No WhatsApp number on supplier page",
        body: "Continuing with Made-in-China inquiry channel.",
      }),
    );
  }

  // 3) WeChat if ID found
  if (contacts?.wechat) {
    const wx = await sendWeChat({ wechatId: contacts.wechat, body: message });
    outreach.push({
      ok: wx.ok,
      channel: "wechat",
      listingUrl: listing.url,
      message,
      wechatId: wx.wechatId,
      error: wx.error,
      sentAt: wx.sentAt,
      contactPerson: contacts.contactPerson,
      supplierName: listing.supplierName,
    });
    timeline.push(
      event({
        actor: "agent",
        kind: wx.ok ? "asked" : "status",
        title: wx.ok
          ? "Agent messaged supplier on WeChat"
          : "WeChat ID captured — queued for ops",
        body: wx.ok
          ? `WeCom message to ${wx.wechatId}.`
          : `${wx.error} WeChat ID: ${wx.wechatId}`,
        facts: { channel: "wechat", wechat_id: wx.wechatId },
      }),
    );
    if (wx.ok) {
      updated = appendOutboundThread(updated, {
        at: wx.sentAt,
        channel: "wechat",
        body: message,
        meta: { wechat_id: wx.wechatId },
      });
    }
  }

  const anyOk = outreach.some((o) => o.ok);
  timeline.push(
    event({
      actor: "system",
      kind: "status",
      title: anyOk
        ? "Outreach sent — awaiting supplier reply"
        : "Outreach incomplete",
      body: anyOk
        ? `Replies: poll agent inbox, email webhook, or paste into the supplier thread. Client never chats directly.`
        : "No channel confirmed delivery. Check MIC/WhatsApp/WeChat configuration.",
    }),
  );

  // Ensure supplier_thread module exists
  let blueprint = updated.blueprint;
  if (!blueprint.modules.some((m) => m.type === "supplier_thread")) {
    const mods = [...blueprint.modules];
    const idx = mods.findIndex((m) => m.type === "negotiation_timeline");
    mods.splice(idx >= 0 ? idx : mods.length - 1, 0, {
      type: "supplier_thread",
      title: "Supplier thread",
    });
    blueprint = { ...blueprint, modules: mods };
  }

  return {
    ...updated,
    status: anyOk ? "awaiting_reply" : "negotiating",
    providerName: mic.supplierName || updated.providerName,
    outreach,
    timeline,
    blueprint,
    contacts: contacts || updated.contacts,
  };
}

