import { categoryLabel } from "./categories";
import type {
  DealRoomBlueprint,
  DealRoomModule,
  HardGoodsCategory,
} from "./types";

export interface BlueprintContext {
  title: string;
  description: string;
  category: HardGoodsCategory;
  quantity: number;
  destinationCountry: string;
  providerCity: string;
  confidence: number;
  dims?: { width: number; depth: number; height: number };
  cbm: number;
  finishes?: { name: string; hex: string; selected?: boolean }[];
}

function layoutFor(
  category: HardGoodsCategory,
): DealRoomBlueprint["layout"] {
  switch (category) {
    case "sofa":
      return "sofa_atelier";
    case "lighting":
      return "lamp_gallery";
    case "table":
      return "table_workshop";
    default:
      return "generic_trade";
  }
}

function accentFor(category: HardGoodsCategory): string {
  switch (category) {
    case "sofa":
      return "#2f6f5e";
    case "lighting":
      return "#b08d57";
    case "table":
      return "#6b4f3a";
    case "storage":
      return "#3d5a80";
    case "outdoor":
      return "#4a7c59";
    case "decor":
      return "#9a5b4a";
    default:
      return "#2f6f5e";
  }
}

/**
 * Generative deal-room coder.
 * Produces a tailored module layout per product — not a single generic template.
 * Swap/extend with an LLM later; rules keep the demo deterministic and offline-capable.
 */
export function generateDealRoomBlueprint(
  ctx: BlueprintContext,
): DealRoomBlueprint {
  const layout = layoutFor(ctx.category);
  const accent = accentFor(ctx.category);
  const modules: DealRoomModule[] = [];

  modules.push({
    type: "hero_stage",
    eyebrow: `${categoryLabel(ctx.category)} · ${ctx.providerCity}`,
    title: ctx.title,
    subtitle:
      ctx.category === "sofa"
        ? "Atelier-style deal room coded for upholstery: dimensions, covers, and cubic freight."
        : ctx.category === "lighting"
          ? "Gallery deal room coded for fixtures: materials, electrical notes, and certs."
          : ctx.category === "table"
            ? "Workshop deal room coded for surfaces: timber, seating count, and pack volume."
            : "Trade deal room coded on the spot for this hard-goods brief.",
    badges: [
      "Hard goods",
      "No sampling",
      "No direct factory chat",
      `${Math.round(ctx.confidence * 100)}% match`,
    ],
    mood:
      ctx.category === "sofa"
        ? "atelier"
        : ctx.category === "lighting"
          ? "gallery"
          : ctx.category === "table"
            ? "workshop"
            : "industrial",
  });

  modules.push({
    type: "match_evidence",
    title: "Producer match",
    confidence: ctx.confidence,
    points:
      ctx.category === "sofa"
        ? [
            "Visual match on arm profile and seat depth cues",
            "Foshan upholstery cluster · export packaging experience",
            "Agent confirmed unbranded style-similar build (no logos)",
          ]
        : ctx.category === "lighting"
          ? [
              "Shade geometry and canopy hardware aligned",
              "Zhongshan lighting corridor supplier",
              "Willing to wire for destination voltage",
            ]
          : [
              "Form factor and material family aligned",
              "Export-ready carpentry / metalwork shop",
              "Photo evidence requested before deposit release",
            ],
  });

  if (ctx.category === "sofa" || ctx.category === "table" || ctx.dims) {
    const d = ctx.dims ?? { width: 220, depth: 95, height: 78 };
    modules.push({
      type: "dimension_figure",
      title: "Spec geometry",
      unit: "cm",
      width: d.width,
      depth: d.depth,
      height: d.height,
      callouts:
        ctx.category === "sofa"
          ? ["Seat depth negotiated", "Arm width locked", "Leg height ±1cm"]
          : ctx.category === "table"
            ? ["Apron clearance", "Top thickness", "Seating long-side"]
            : ["Overall envelope for freight"],
    });
  }

  if (ctx.category === "sofa" || ctx.category === "decor") {
    modules.push({
      type: "finish_strip",
      title: ctx.category === "sofa" ? "Cover options" : "Finish options",
      options: ctx.finishes ?? [
        { name: "Natural linen", hex: "#d8d0c4", selected: true },
        { name: "Stone bouclé", hex: "#b7b1a8" },
        { name: "Ink performance", hex: "#2c3338" },
        { name: "Clay weave", hex: "#a66a4e" },
      ],
      note: "Selections are confirmed in the agent↔factory thread; you only approve here.",
    });
  }

  if (ctx.category === "lighting" || ctx.category === "decor") {
    modules.push({
      type: "material_callouts",
      title: "Materials",
      materials:
        ctx.category === "lighting"
          ? [
              { name: "Shade", detail: "Hand-pleated fabric over steel ring" },
              { name: "Canopy", detail: "Brushed brass, ceiling mount" },
              { name: "Diffuser", detail: "Opal glass · replaceable" },
            ]
          : [
              { name: "Body", detail: "Glazed ceramic · export crate foam" },
              { name: "Base", detail: "Felted foot pads included" },
            ],
    });
  }

  if (ctx.category === "lighting") {
    modules.push({
      type: "electrical_notes",
      title: "Electrical",
      voltage: ctx.destinationCountry.toUpperCase().startsWith("US")
        ? "120V / E26"
        : "220–240V / E27",
      certs: ["CE mark on driver path", "UL-style components on request"],
      notes: [
        "Hardwired canopy · dimmer compatibility TBD in next agent ping",
        "No live bulbs in carton · socket only",
      ],
    });
  }

  if (ctx.category === "table" || ctx.category === "storage") {
    modules.push({
      type: "material_callouts",
      title: "Build",
      materials: [
        { name: "Primary", detail: "Solid oak top · mortise apron" },
        {
          name: "Finish",
          detail: "Matte hardwax oil · sample chip photo on QC",
        },
        { name: "Hardware", detail: "Hidden levelers · felt tips" },
      ],
    });
  }

  modules.push({
    type: "spec_grid",
    title: "Locked commercial brief",
    items: [
      { label: "Quantity", value: String(ctx.quantity) },
      { label: "Destination", value: ctx.destinationCountry },
      { label: "Category", value: categoryLabel(ctx.category) },
      {
        label: "Client channel",
        value: "Deal room only",
        hint: "No WhatsApp / WeChat exposed",
      },
    ],
  });

  modules.push({
    type: "freight_volume",
    title: "Pack & lane",
    cbm: ctx.cbm,
    packNote:
      ctx.category === "sofa"
        ? "Knock-down legs · blanket wrap · corner boards"
        : ctx.category === "lighting"
          ? "Double-wall carton · glass sleeve · humidity pack"
          : "Export crate · desiccant · shock label",
    route: `${ctx.providerCity} → ${ctx.destinationCountry} (sea, DDP estimate)`,
  });

  modules.push({
    type: "risk_flags",
    title: "Watchouts",
    flags:
      ctx.category === "sofa"
        ? [
            {
              level: "watch",
              text: "Fabric dye lot may shift ±1 shade — approve QC swatch photo",
            },
            {
              level: "info",
              text: "Style-similar unbranded build; not an authorized brand replica",
            },
          ]
        : ctx.category === "lighting"
          ? [
              {
                level: "watch",
                text: "Confirm canopy hole spacing against your junction box",
              },
              {
                level: "info",
                text: "Bulbs excluded to simplify customs description",
              },
            ]
          : [
              {
                level: "info",
                text: "Wood moisture content checked at QC before crate close",
              },
            ],
  });

  modules.push({
    type: "negotiation_timeline",
    title: "Negotiation retranscription",
  });
  modules.push({ type: "landed_quote", title: "Landed quote" });
  modules.push({ type: "actions", title: "Your move" });

  return {
    generatedAt: new Date().toISOString(),
    generator: "rules",
    category: ctx.category,
    layout,
    accent,
    modules,
  };
}
