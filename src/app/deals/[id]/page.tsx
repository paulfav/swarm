import { DealRoom } from "@/components/deal-room";
import { getDeal } from "@/lib/store";
import { notFound } from "next/navigation";

export default async function DealPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const deal = getDeal(id);
  if (!deal) notFound();
  return <DealRoom initialDeal={deal} />;
}
