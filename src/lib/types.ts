export type DealStatus =
  | "sourcing"
  | "negotiating"
  | "quoted"
  | "deposit_due"
  | "in_production"
  | "qc"
  | "shipped"
  | "closed"
  | "rejected";

export type HardGoodsCategory =
  | "sofa"
  | "lighting"
  | "table"
  | "storage"
  | "outdoor"
  | "decor"
  | "other_hard_good";

export type TimelineActor = "agent" | "factory" | "system" | "client";

export interface TimelineEvent {
  id: string;
  at: string;
  actor: TimelineActor;
  kind:
    | "sourced"
    | "asked"
    | "answered"
    | "pushed"
    | "agreed"
    | "risk"
    | "spec_locked"
    | "status";
  title: string;
  body: string;
  facts?: Record<string, string>;
}

export interface QuoteLine {
  label: string;
  amountUsd: number;
  note?: string;
}

export interface Quote {
  version: number;
  currency: "USD";
  factoryPriceUsd: number;
  lines: QuoteLine[];
  landedTotalUsd: number;
  leadTimeDays: number;
  moq: number;
  incoterm: "FOB" | "EXW" | "DDP";
  validUntil: string;
  guaranteeSummary: string;
}

export interface SpecLock {
  label: string;
  value: string;
}

/** Generative deal-room blueprint — AI-coded per product */
export type DealRoomModule =
  | {
      type: "hero_stage";
      eyebrow: string;
      title: string;
      subtitle: string;
      badges: string[];
      mood: "atelier" | "industrial" | "gallery" | "workshop";
    }
  | {
      type: "spec_grid";
      title: string;
      items: { label: string; value: string; hint?: string }[];
    }
  | {
      type: "dimension_figure";
      title: string;
      unit: "cm" | "in";
      width: number;
      depth: number;
      height: number;
      callouts: string[];
    }
  | {
      type: "finish_strip";
      title: string;
      options: { name: string; hex: string; selected?: boolean }[];
      note?: string;
    }
  | {
      type: "material_callouts";
      title: string;
      materials: { name: string; detail: string }[];
    }
  | {
      type: "freight_volume";
      title: string;
      cbm: number;
      packNote: string;
      route: string;
    }
  | {
      type: "electrical_notes";
      title: string;
      voltage: string;
      certs: string[];
      notes: string[];
    }
  | {
      type: "risk_flags";
      title: string;
      flags: { level: "info" | "watch" | "critical"; text: string }[];
    }
  | {
      type: "match_evidence";
      title: string;
      confidence: number;
      points: string[];
    }
  | {
      type: "negotiation_timeline";
      title: string;
    }
  | {
      type: "landed_quote";
      title: string;
    }
  | {
      type: "actions";
      title: string;
    };

export interface DealRoomBlueprint {
  generatedAt: string;
  generator: "rules" | "llm";
  category: HardGoodsCategory;
  layout: "sofa_atelier" | "lamp_gallery" | "table_workshop" | "generic_trade";
  accent: string;
  modules: DealRoomModule[];
}

export interface InquiryInput {
  title?: string;
  description: string;
  imageDataUrl?: string;
  sourceUrl?: string;
  quantity: number;
  destinationCountry: string;
  budgetUsd?: number;
}

export interface Deal {
  id: string;
  createdAt: string;
  updatedAt: string;
  status: DealStatus;
  category: HardGoodsCategory;
  title: string;
  description: string;
  imageDataUrl?: string;
  sourceUrl?: string;
  quantity: number;
  destinationCountry: string;
  budgetUsd?: number;
  providerName: string;
  providerCity: string;
  blueprint: DealRoomBlueprint;
  timeline: TimelineEvent[];
  quote: Quote;
  specLock: SpecLock[];
}
