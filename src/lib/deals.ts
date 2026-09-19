import { classifyHardGood } from "./categories";
import { generateDealRoomBlueprint } from "./generate-deal-room";
import { buildLandedQuote } from "./landed-cost";
import type {
  Deal,
  DealStatus,
  HardGoodsCategory,
  InquiryInput,
  TimelineEvent,
} from "./types";

function id(prefix: string): string {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`;
}

function titleFromInput(input: InquiryInput, category: HardGoodsCategory): string {
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

function providerFor(category: HardGoodsCategory): {
  name: string;
  city: string;
} {
  switch (category) {
    case "sofa":
      return { name: "Foshan Lianyu Upholstery", city: "Foshan" };
    case "lighting":
      return { name: "Zhongshan Helm Lighting", city: "Zhongshan" };
    case "table":
      return { name: "Dongguan Muir Woodworks", city: "Dongguan" };
    case "storage":
      return { name: "Shunde Cabinet Works", city: "Shunde" };
    default:
      return { name: "Shenzhen Export Collective", city: "Shenzhen" };
  }
}

function economics(category: HardGoodsCategory, quantity: number) {
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
}

function buildTimeline(
  category: HardGoodsCategory,
  provider: { name: string; city: string },
  factoryPrice: number,
  moq: number,
): TimelineEvent[] {
  const now = Date.now();
  const at = (minsAgo: number) => new Date(now - minsAgo * 60_000).toISOString();

  const base: TimelineEvent[] = [
    {
      id: id("evt"),
      at: at(120),
      actor: "system",
      kind: "sourced",
      title: "Producer shortlisted",
      body: `Agent matched ${provider.name} (${provider.city}) from export-capable hard-goods graph.`,
      facts: { channel: "WeChat (internal)", language: "zh-CN" },
    },
    {
      id: id("evt"),
      at: at(100),
      actor: "agent",
      kind: "asked",
      title: "Agent asked for FOB + lead time",
      body: "Sent reference photos, target dimensions, destination, and quantity. Requested FOB, MOQ, and production calendar.",
    },
    {
      id: id("evt"),
      at: at(85),
      actor: "factory",
      kind: "answered",
      title: "Factory replied with first quote",
      body:
        category === "sofa"
          ? `Quoted ¥${Math.round(factoryPrice * 7.2)} FOB for linen cover, MOQ ${Math.max(moq, 5)}, 18–20 days after deposit.`
          : `Quoted ¥${Math.round(factoryPrice * 7.2)} FOB, MOQ ${Math.max(moq, 5)}, standard export pack.`,
      facts: {
        moq: String(Math.max(moq, 5)),
        lead_time: "18–20 days",
      },
    },
    {
      id: id("evt"),
      at: at(70),
      actor: "agent",
      kind: "pushed",
      title: "Agent pushed MOQ and unit economics",
      body: `Asked to honor MOQ ${moq} for a single high-ticket export order and remove retail-middle markup assumptions.`,
    },
    {
      id: id("evt"),
      at: at(40),
      actor: "factory",
      kind: "agreed",
      title: "Factory agreed revised terms",
      body: `Accepted MOQ ${moq} with minor unit uplift. Confirmed unbranded build and pre-ship photo QC.`,
      facts: {
        moq: String(moq),
        fob_usd: `$${factoryPrice}`,
      },
    },
    {
      id: id("evt"),
      at: at(15),
      actor: "system",
      kind: "status",
      title: "Retranscription published to deal room",
      body: "Raw Chinese thread kept internal. Client sees structured cards + landed quote only.",
    },
  ];

  if (category === "lighting") {
    base.splice(4, 0, {
      id: id("evt"),
      at: at(55),
      actor: "agent",
      kind: "asked",
      title: "Agent confirmed voltage & canopy",
      body: "Requested 120V socket path and canopy drill template photo before deposit.",
    });
  }

  return base;
}

export function createDealFromInquiry(input: InquiryInput): Deal {
  const blob = `${input.title ?? ""} ${input.description} ${input.sourceUrl ?? ""}`;
  const category = classifyHardGood(blob);
  const title = titleFromInput(input, category);
  const provider = providerFor(category);
  const eco = economics(category, input.quantity);
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
    confidence: eco.confidence,
    dims: eco.dims,
    cbm: eco.cbm,
  });

  const now = new Date().toISOString();
  return {
    id: id("deal"),
    createdAt: now,
    updatedAt: now,
    status: "quoted",
    category,
    title,
    description: input.description,
    imageDataUrl: input.imageDataUrl,
    sourceUrl: input.sourceUrl,
    quantity: input.quantity,
    destinationCountry: input.destinationCountry,
    budgetUsd: input.budgetUsd,
    providerName: provider.name,
    providerCity: provider.city,
    blueprint,
    timeline: buildTimeline(category, provider, eco.factory, eco.moq),
    quote,
    specLock: [
      { label: "Product", value: title },
      { label: "Origin", value: `${provider.city}, China` },
      { label: "Incoterm path", value: "Factory FOB → platform DDP to client" },
      { label: "Sampling", value: "None — hard goods photo/spec lock only" },
    ],
  };
}

export function applyDealAction(
  deal: Deal,
  action: "approve" | "request_change" | "reject",
  note?: string,
): Deal {
  const now = new Date().toISOString();
  const event = (partial: Omit<TimelineEvent, "id" | "at">): TimelineEvent => ({
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
          body: "Deposit gate opened. Factory will not see client identity — agent continues the thread.",
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

  // request_change — agent goes back to factory; bump quote slightly as demo
  const newFactory = Math.max(
    40,
    Math.round(deal.quote.factoryPriceUsd * 0.94),
  );
  const quote = buildLandedQuote({
    factoryPriceUsd: newFactory,
    cbm: deal.blueprint.modules.find((m) => m.type === "freight_volume")
      ?.type === "freight_volume"
      ? (
          deal.blueprint.modules.find((m) => m.type === "freight_volume") as {
            cbm: number;
          }
        ).cbm
      : 0.5,
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
        title: "Agent re-opened factory thread",
        body: "Translated the request into Chinese commercial language and countered.",
      }),
      event({
        actor: "factory",
        kind: "agreed",
        title: "Factory sent revised FOB",
        body: `New FOB $${newFactory}. Deal room quote refreshed to v${quote.version}.`,
        facts: { fob_usd: `$${newFactory}`, quote_version: String(quote.version) },
      }),
    ],
  };
}
