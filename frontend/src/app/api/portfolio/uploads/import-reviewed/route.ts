import { NextResponse } from "next/server";

import {
  getBackendHeaders,
  INTERNAL_API_BASE_URL,
  parseBackendResponse,
} from "@/lib/server-api";

export async function POST(request: Request) {
  const body = await request.json();

  const response = await fetch(
    `${INTERNAL_API_BASE_URL}/portfolio/uploads/import-reviewed`,
    {
      method: "POST",
      headers: getBackendHeaders({
        "Content-Type": "application/json",
      }),
      body: JSON.stringify(body),
      cache: "no-store",
    }
  );

  const data = await parseBackendResponse(response);

  return NextResponse.json(data, {
    status: response.status,
  });
}
