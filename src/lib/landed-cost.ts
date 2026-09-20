import type { Quote, QuoteLine } from "./types";

export function buildLandedQuote(input: {
  factoryPriceUsd: number;
  cbm: number;
  destinationCountry: string;
  leadTimeDays: number;
  moq: number;
  version?: number;
}): Quote {
  const freight = Math.max(90, Math.round(input.cbm * 180));
  const dutyRate =
    input.destinationCountry.toUpperCase() === "US" ||
    input.destinationCountry.toUpperCase() === "USA"
      ? 0.18
      : 0.12;
  const duties = Math.round(input.factoryPriceUsd * dutyRate);
  const brokerage = 75;
  const insurance = Math.round((input.factoryPriceUsd + freight) * 0.015);
  const platform = Math.round(
    (input.factoryPriceUsd + freight + duties) * 0.14,
  );
  const qc = input.factoryPriceUsd > 400 ? 65 : 0;

  const lines: QuoteLine[] = [
    {
      label: "Factory (FOB)",
      amountUsd: input.factoryPriceUsd,
      note: "Negotiated by agent",
    },
    {
      label: "Ocean freight + inland",
      amountUsd: freight,
      note: `${input.cbm.toFixed(2)} CBM estimate`,
    },
    {
      label: "Duties & tariffs",
      amountUsd: duties,
      note: `${Math.round(dutyRate * 100)}% estimate · ${input.destinationCountry}`,
    },
    { label: "Brokerage / clearance", amountUsd: brokerage },
    { label: "Cargo insurance", amountUsd: insurance },
  ];
  if (qc) {
    lines.push({
      label: "Pre-ship QC photos",
      amountUsd: qc,
      note: "Required before balance",
    });
  }
  lines.push({
    label: "China Access fee",
    amountUsd: platform,
    note: "Sourcing, negotiation, translation, guarantee desk",
  });

  const landedTotalUsd = lines.reduce((s, l) => s + l.amountUsd, 0);
  const valid = new Date();
  valid.setDate(valid.getDate() + 7);

  return {
    version: input.version ?? 1,
    currency: "USD",
    factoryPriceUsd: input.factoryPriceUsd,
    lines,
    landedTotalUsd,
    leadTimeDays: input.leadTimeDays,
    moq: input.moq,
    incoterm: "DDP",
    validUntil: valid.toISOString(),
    guaranteeSummary:
      "Spec-lock match · transit damage insured · QC photo gate before ship · 30-day structural defect window",
  };
}
