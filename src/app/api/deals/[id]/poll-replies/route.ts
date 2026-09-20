import { NextResponse } from "next/server";
import { pollAndIngestReplies } from "@/lib/replies";
import { getDealAsync, saveDeal } from "@/lib/store";

export const maxDuration = 120;

export async function POST(
  _request: Request,
  context: { params: Promise<{ id: string }> },
) {
  const { id } = await context.params;
  const deal = await getDealAsync(id);
  if (!deal) {
    return NextResponse.json({ error: "Deal not found" }, { status: 404 });
  }

  const result = await pollAndIngestReplies(deal);
  saveDeal(result.deal);
  return NextResponse.json(result);
}
