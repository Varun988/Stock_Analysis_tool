import { NextRequest, NextResponse } from "next/server";

import {
  getBackendHeaders,
  INTERNAL_API_BASE_URL,
  parseBackendResponse,
} from "@/lib/server-api";

async function proxyInstrumentRequest(
  method: "GET" | "POST",
  body?: unknown,
) {
  const response = await fetch(`${INTERNAL_API_BASE_URL}/instruments`, {
    method,
    headers: getBackendHeaders({
      "Content-Type": "application/json",
    }),
    body: body ? JSON.stringify(body) : undefined,
    cache: "no-store",
  });

  const data = await parseBackendResponse(response);

  return NextResponse.json(data, {
    status: response.status,
  });
}

export async function GET() {
  return proxyInstrumentRequest("GET");
}

export async function POST(request: NextRequest) {
  const body = await request.json();
  return proxyInstrumentRequest("POST", body);
}