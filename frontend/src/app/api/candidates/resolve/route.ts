import { NextResponse } from "next/server";

import {
  getBackendHeaders,
  INTERNAL_API_BASE_URL,
  parseBackendResponse,
} from "@/lib/server-api";

export async function POST(request: Request) {
  const payload = await request.json();

  const response = await fetch(`${INTERNAL_API_BASE_URL}/candidates/resolve`, {
    method: "POST",
    headers: getBackendHeaders({
      "Content-Type": "application/json",
    }),
    body: JSON.stringify(payload),
    cache: "no-store",
  });

  const data = await parseBackendResponse(response);

  return NextResponse.json(data, {
    status: response.status,
  });
}
