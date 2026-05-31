import { NextResponse } from "next/server";

import {
  getBackendHeaders,
  INTERNAL_API_BASE_URL,
  parseBackendResponse,
} from "@/lib/server-api";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const useLlmSummary = searchParams.get("use_llm_summary") ?? "true";

  const response = await fetch(
    `${INTERNAL_API_BASE_URL}/research/market/india?use_llm_summary=${useLlmSummary}`,
    {
      method: "GET",
      headers: getBackendHeaders(),
      cache: "no-store",
    }
  );

  const data = await parseBackendResponse(response);

  return NextResponse.json(data, {
    status: response.status,
  });
}