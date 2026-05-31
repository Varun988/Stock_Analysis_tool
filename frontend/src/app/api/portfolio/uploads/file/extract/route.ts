import { NextResponse } from "next/server";

import {
  getBackendHeaders,
  INTERNAL_API_BASE_URL,
  parseBackendResponse,
} from "@/lib/server-api";

export async function POST(request: Request) {
  const formData = await request.formData();

  const response = await fetch(
    `${INTERNAL_API_BASE_URL}/portfolio/uploads/file/extract`,
    {
      method: "POST",
      headers: getBackendHeaders(),
      body: formData,
      cache: "no-store",
    }
  );

  const data = await parseBackendResponse(response);

  return NextResponse.json(data, {
    status: response.status,
  });
}