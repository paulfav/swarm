import { createDealFromInquiry } from "./deals";
import type { Deal } from "./types";

/**
 * Simple in-memory store for the MVP.
 * Survives across hot reloads via globalThis in dev.
 */
const globalForStore = globalThis as unknown as {
  __chinaAccessDeals?: Map<string, Deal>;
};

function store(): Map<string, Deal> {
  if (!globalForStore.__chinaAccessDeals) {
    globalForStore.__chinaAccessDeals = new Map();
    seed(globalForStore.__chinaAccessDeals);
  }
  return globalForStore.__chinaAccessDeals;
}

function seed(map: Map<string, Deal>) {
  const seeds = [
    createDealFromInquiry({
      title: "Cloud-style 3-seat linen sofa",
      description:
        "Looking for a deep-seat linen sofa similar to a high-end US cloud couch. Soft arms, 220cm wide, performance linen, shipped to USA.",
      quantity: 1,
      destinationCountry: "USA",
      budgetUsd: 900,
    }),
    createDealFromInquiry({
      title: "Brass pendant with pleated shade",
      description:
        "Gallery pendant lamp, brushed brass canopy, hand-pleated fabric shade, hardwired. Need US voltage.",
      quantity: 2,
      destinationCountry: "USA",
      budgetUsd: 260,
    }),
    createDealFromInquiry({
      title: "Oak dining table for eight",
      description:
        "Solid oak dining table 200cm, matte oil finish, seating for 8, clean workshop aesthetic.",
      quantity: 1,
      destinationCountry: "USA",
      budgetUsd: 1100,
    }),
  ];

  // Stable demo IDs
  const ids = ["deal_sofa_demo", "deal_lamp_demo", "deal_table_demo"];
  seeds.forEach((deal, i) => {
    deal.id = ids[i]!;
    map.set(deal.id, deal);
  });
}

export function listDeals(): Deal[] {
  return Array.from(store().values()).sort((a, b) =>
    b.updatedAt.localeCompare(a.updatedAt),
  );
}

export function getDeal(id: string): Deal | undefined {
  return store().get(id);
}

export function saveDeal(deal: Deal): Deal {
  store().set(deal.id, deal);
  return deal;
}
