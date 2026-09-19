import { DealRoom } from "@/components/deal-room";
import { getDealAsync } from "@/lib/store";
import { notFound } from "next/navigation";

export const dynamic = "force-dynamic";
export const maxDuration = 120;

export default async function DealPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const deal = await getDealAsync(id);
  if (!deal) notFound();
  return <DealRoom initialDeal={deal} />;
}
