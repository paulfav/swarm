import { NextResponse } from "next/server";
import { createDealFromInquiry } from "@/lib/deals";
import { listDealsAsync, saveDeal } from "@/lib/store";
import type { InquiryInput } from "@/lib/types";

export const maxDuration = 120;

export async function GET() {
  const deals = await listDealsAsync();
  return NextResponse.json({ deals });
}

export async function POST(request: Request) {
  const body = (await request.json()) as Partial<InquiryInput> & {
    skipLiveSource?: boolean;
  };
  if (!body.description?.trim()) {
    return NextResponse.json(
      { error: "description is required" },
      { status: 400 },
    );
  }

  const deal = await createDealFromInquiry(
    {
      title: body.title,
      description: body.description,
      imageDataUrl: body.imageDataUrl,
      sourceUrl: body.sourceUrl,
      quantity: Math.max(1, Number(body.quantity) || 1),
      destinationCountry: body.destinationCountry?.trim() || "USA",
      budgetUsd: body.budgetUsd ? Number(body.budgetUsd) : undefined,
    },
    { skipLiveSource: Boolean(body.skipLiveSource) },
  );

  saveDeal(deal);
  return NextResponse.json({ deal }, { status: 201 });
}
