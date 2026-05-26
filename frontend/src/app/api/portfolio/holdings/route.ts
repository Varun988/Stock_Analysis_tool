import { NextRequest, NextResponse } from "next/server";

const API_BASE_URL =
  process.env.INTERNAL_API_BASE_URL ?? "http://localhost:8000/api/v1";

async function proxyPortfolioHoldingsRequest(
  method: "GET" | "POST",
  body?: unknown,
) {
  const response = await fetch(`${API_BASE_URL}/portfolio/holdings`, {
    method,
    headers: {
      "Content-Type": "application/json",
    },
    body: body ? JSON.stringify(body) : undefined,
    cache: "no-store",
  });

  const data = await response.json();

  return NextResponse.json(data, {
    status: response.status,
  });
}

export async function GET() {
  return proxyPortfolioHoldingsRequest("GET");
}

export async function POST(request: NextRequest) {
  const body = await request.json();
  return proxyPortfolioHoldingsRequest("POST", body);
}