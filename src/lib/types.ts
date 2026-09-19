export type DealStatus =
  | "sourcing"
  | "negotiating"
  | "quoted"
  | "deposit_due"
  | "in_production"
  | "qc"
  | "shipped"
  | "closed"
  | "rejected"
  | "awaiting_reply";

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
      type: "live_listings";
      title: string;
    }
  | {
      type: "supplier_thread";
      title: string;
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

export interface SourcedListingView {
  source: "made-in-china" | "aliexpress" | "alibaba";
  title: string;
  url: string;
  priceUsd?: number;
  imageUrl?: string;
  supplierName?: string;
  supplierUrl?: string;
  rawPriceText?: string;
}

export interface SourceAttemptView {
  source: SourcedListingView["source"];
  ok: boolean;
  status: "ok" | "captcha" | "empty" | "error";
  detail: string;
  query: string;
  searchedAt: string;
  listingCount: number;
}

export type OutreachChannel =
  | "made-in-china-inquiry"
  | "whatsapp"
  | "wechat"
  | "email";

export interface OutreachRecord {
  ok: boolean;
  channel: OutreachChannel;
  supplierName?: string;
  contactPerson?: string;
  listingUrl?: string;
  message: string;
  identityEmail?: string;
  successUrl?: string;
  inquiryId?: string;
  whatsappTo?: string;
  wechatId?: string;
  deepLink?: string;
  steps?: string[];
  error?: string;
  sentAt: string;
}

export interface ThreadMessage {
  id: string;
  at: string;
  direction: "outbound" | "inbound";
  channel: OutreachChannel | "system";
  author: "agent" | "supplier" | "system";
  body: string;
  retranscription?: string;
  meta?: Record<string, string>;
}

export interface SupplierContactInfo {
  whatsapp?: string;
  wechat?: string;
  email?: string;
  phone?: string;
  contactPerson?: string;
  pageUrl?: string;
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
  sourcing?: {
    live: boolean;
    query: string;
    attempts: SourceAttemptView[];
    listings: SourcedListingView[];
  };
  contacts?: SupplierContactInfo;
  outreach?: OutreachRecord[];
  thread?: ThreadMessage[];
  blueprint: DealRoomBlueprint;
  timeline: TimelineEvent[];
  quote: Quote;
  specLock: SpecLock[];
}
