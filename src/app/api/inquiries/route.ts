import { NextResponse } from "next/server";
import { createDealFromInquiry } from "@/lib/deals";
import { listDeals, saveDeal } from "@/lib/store";
import type { InquiryInput } from "@/lib/types";

export async function GET() {
  return NextResponse.json({ deals: listDeals() });
}

export async function POST(request: Request) {
  const body = (await request.json()) as Partial<InquiryInput>;
  if (!body.description?.trim()) {
    return NextResponse.json(
      { error: "description is required" },
      { status: 400 },
    );
  }

  const deal = createDealFromInquiry({
    title: body.title,
    description: body.description,
    imageDataUrl: body.imageDataUrl,
    sourceUrl: body.sourceUrl,
    quantity: Math.max(1, Number(body.quantity) || 1),
    destinationCountry: body.destinationCountry?.trim() || "USA",
    budgetUsd: body.budgetUsd ? Number(body.budgetUsd) : undefined,
  });

  saveDeal(deal);
  return NextResponse.json({ deal }, { status: 201 });
}
