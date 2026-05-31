import { NextResponse } from "next/server";

import {
  getBackendHeaders,
  INTERNAL_API_BASE_URL,
  parseBackendResponse,
} from "@/lib/server-api";

export async function GET() {
  const response = await fetch(
    `${INTERNAL_API_BASE_URL}/explanations/history`,
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