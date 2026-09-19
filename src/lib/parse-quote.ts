import type { Quote } from "./types";
import { buildLandedQuote } from "./landed-cost";

export interface ParsedSupplierQuote {
  factoryPriceUsd?: number;
  moq?: number;
  leadTimeDays?: number;
  cbm?: number;
  currencyNote?: string;
  rawSnippets: string[];
}

/** Pull commercial facts from a supplier reply (EN or CN mixed). */
export function parseSupplierReply(text: string): ParsedSupplierQuote {
  const rawSnippets: string[] = [];
  const out: ParsedSupplierQuote = { rawSnippets };

  const usd =
    text.match(
      /(?:FOB|EXW|CIF|price|单价|报价)[^\d$¥￥]{0,20}(?:US\$|USD|\$)\s*([0-9]{2,6}(?:\.[0-9]+)?)/i,
    ) ||
    text.match(/(?:US\$|USD|\$)\s*([0-9]{2,6}(?:\.[0-9]+)?)\s*(?:\/\s*(?:pc|unit|set|件))?/i);
  if (usd) {
    out.factoryPriceUsd = Number(usd[1]);
    rawSnippets.push(usd[0]);
  } else {
    const cny = text.match(
      /(?:FOB|EXW|价格|报价|单价)[^\d¥￥]{0,12}[¥￥]\s*([0-9]{2,7}(?:\.[0-9]+)?)/i,
    );
    if (cny) {
      out.factoryPriceUsd = Math.round(Number(cny[1]) / 7.2);
      out.currencyNote = `Parsed ¥${cny[1]} ≈ $${out.factoryPriceUsd} @ 7.2`;
      rawSnippets.push(cny[0]);
    }
  }

  const moq = text.match(
    /(?:MOQ|最小订量|起订量)[^\d]{0,12}(\d{1,5})/i,
  );
  if (moq) {
    out.moq = Number(moq[1]);
    rawSnippets.push(moq[0]);
  }

  const lead = text.match(
    /(?:lead\s*time|交期|工期|生产周期)[^\d]{0,16}(\d{1,3})\s*(?:day|days|天|工作日)?/i,
  );
  if (lead) {
    out.leadTimeDays = Number(lead[1]);
    rawSnippets.push(lead[0]);
  }

  const cbm = text.match(/(?:CBM|体积|立方)[^\d]{0,10}(\d+(?:\.\d+)?)/i);
  if (cbm) {
    out.cbm = Number(cbm[1]);
    rawSnippets.push(cbm[0]);
  }

  return out;
}

export function applyParsedQuoteToDealQuote(
  current: Quote,
  parsed: ParsedSupplierQuote,
  destinationCountry: string,
): Quote | null {
  if (
    parsed.factoryPriceUsd == null &&
    parsed.moq == null &&
    parsed.leadTimeDays == null &&
    parsed.cbm == null
  ) {
    return null;
  }

  const freightModuleCbm = parsed.cbm;
  return buildLandedQuote({
    factoryPriceUsd: parsed.factoryPriceUsd ?? current.factoryPriceUsd,
    cbm: freightModuleCbm ?? Math.max(0.2, current.factoryPriceUsd / 400),
    destinationCountry,
    leadTimeDays: parsed.leadTimeDays ?? current.leadTimeDays,
    moq: parsed.moq ?? current.moq,
    version: current.version + 1,
  });
}

/** Translate-ish summary for the client deal room (EN). */
export function retranscribeReply(text: string, parsed: ParsedSupplierQuote): string {
  const bits: string[] = ["Supplier replied (retranscription):"];
  if (parsed.factoryPriceUsd != null) {
    bits.push(`• Factory/FOB signal ≈ $${parsed.factoryPriceUsd}${parsed.currencyNote ? ` (${parsed.currencyNote})` : ""}`);
  }
  if (parsed.moq != null) bits.push(`• MOQ ${parsed.moq}`);
  if (parsed.leadTimeDays != null) bits.push(`• Lead time ~${parsed.leadTimeDays} days`);
  if (parsed.cbm != null) bits.push(`• Pack volume ~${parsed.cbm} CBM`);
  if (bits.length === 1) {
    bits.push(text.slice(0, 400));
  } else {
    bits.push("", "Original excerpt:", text.slice(0, 280));
  }
  return bits.join("\n");
}
