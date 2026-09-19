import type { HardGoodsCategory } from "./types";

const SOFA = /\b(sofa|couch|sectional|loveseat|chaise|canape|canapé)\b/i;
const LIGHT =
  /\b(lamp|pendant|sconce|chandelier|lighting|floor lamp|table lamp)\b/i;
const TABLE = /\b(table|desk|console|dining|coffee table)\b/i;
const STORAGE = /\b(cabinet|sideboard|dresser|shelf|shelving|wardrobe|bookcase)\b/i;
const OUTDOOR = /\b(outdoor|patio|garden|teak|lounger)\b/i;
const DECOR = /\b(vase|mirror|frame|sculpture|decor|décor|planter)\b/i;

export function classifyHardGood(text: string): HardGoodsCategory {
  if (SOFA.test(text)) return "sofa";
  if (LIGHT.test(text)) return "lighting";
  if (TABLE.test(text)) return "table";
  if (STORAGE.test(text)) return "storage";
  if (OUTDOOR.test(text)) return "outdoor";
  if (DECOR.test(text)) return "decor";
  return "other_hard_good";
}

export function categoryLabel(category: HardGoodsCategory): string {
  switch (category) {
    case "sofa":
      return "Upholstered seating";
    case "lighting":
      return "Lighting";
    case "table":
      return "Tables & surfaces";
    case "storage":
      return "Storage";
    case "outdoor":
      return "Outdoor hard goods";
    case "decor":
      return "Décor objects";
    default:
      return "Hard goods";
  }
}
