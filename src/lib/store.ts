import { createDealFromInquiry } from "./deals";
import type { Deal } from "./types";

/**
 * Simple in-memory store for the MVP.
 * Survives across hot reloads via globalThis in dev.
 */
const globalForStore = globalThis as unknown as {
  __chinaAccessDeals?: Map<string, Deal>;
  __chinaAccessSeedPromise?: Promise<void>;
};

function store(): Map<string, Deal> {
  if (!globalForStore.__chinaAccessDeals) {
    globalForStore.__chinaAccessDeals = new Map();
  }
  return globalForStore.__chinaAccessDeals;
}

async function seedLiveDemos(map: Map<string, Deal>) {
  const specs = [
    {
      id: "deal_sofa_demo",
      input: {
        title: "Cloud-style 3-seat linen sofa",
        description:
          "Looking for a deep-seat linen sofa similar to a high-end US cloud couch. Soft arms, 220cm wide, performance linen, shipped to USA.",
        quantity: 1,
        destinationCountry: "USA",
        budgetUsd: 900,
      },
    },
    {
      id: "deal_lamp_demo",
      input: {
        title: "Brass pendant with pleated shade",
        description:
          "Gallery pendant lamp, brushed brass canopy, hand-pleated fabric shade, hardwired. Need US voltage.",
        quantity: 2,
        destinationCountry: "USA",
        budgetUsd: 260,
      },
    },
    {
      id: "deal_table_demo",
      input: {
        title: "Oak dining table for eight",
        description:
          "Solid oak dining table 200cm, matte oil finish, seating for 8, clean workshop aesthetic.",
        quantity: 1,
        destinationCountry: "USA",
        budgetUsd: 1100,
      },
    },
  ] as const;

  // Seed placeholders first so UI is never empty while scraping
  for (const spec of specs) {
    if (map.has(spec.id)) continue;
    const placeholder = await createDealFromInquiry(spec.input, {
      skipLiveSource: true,
    });
    placeholder.id = spec.id;
    map.set(spec.id, placeholder);
  }

  // Then replace with live China marketplace results (sequential to be kinder to sites)
  for (const spec of specs) {
    try {
      const live = await createDealFromInquiry(spec.input);
      live.id = spec.id;
      map.set(spec.id, live);
    } catch {
      // keep placeholder
    }
  }
}

export function ensureSeeded(): Promise<void> {
  if (!globalForStore.__chinaAccessSeedPromise) {
    globalForStore.__chinaAccessSeedPromise = seedLiveDemos(store());
  }
  return globalForStore.__chinaAccessSeedPromise;
}

export function listDeals(): Deal[] {
  void ensureSeeded();
  return Array.from(store().values()).sort((a, b) =>
    b.updatedAt.localeCompare(a.updatedAt),
  );
}

export async function listDealsAsync(): Promise<Deal[]> {
  await ensureSeeded();
  return listDeals();
}

export function getDeal(id: string): Deal | undefined {
  void ensureSeeded();
  return store().get(id);
}

export async function getDealAsync(id: string): Promise<Deal | undefined> {
  await ensureSeeded();
  const deal = store().get(id);
  if (!deal) return undefined;
  return ensureThreadModule(deal);
}

function ensureThreadModule(deal: Deal): Deal {
  if (deal.blueprint.modules.some((m) => m.type === "supplier_thread")) {
    return deal;
  }
  const mods = [...deal.blueprint.modules];
  const idx = mods.findIndex((m) => m.type === "actions");
  mods.splice(idx >= 0 ? idx : mods.length, 0, {
    type: "supplier_thread",
    title: "Supplier thread",
  });
  const next = { ...deal, blueprint: { ...deal.blueprint, modules: mods } };
  store().set(deal.id, next);
  return next;
}

export function saveDeal(deal: Deal): Deal {
  store().set(deal.id, deal);
  return deal;
}
