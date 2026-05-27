import { NextResponse } from "next/server";

const API_BASE_URL =
  process.env.INTERNAL_API_BASE_URL ?? "http://localhost:8000/api/v1";

async function parseBackendResponse(response: Response) {
  const text = await response.text();

  if (!text) {
    return {
      detail: `Backend returned empty response with status ${response.status}`,
    };
  }

  try {
    return JSON.parse(text);
  } catch {
    return {
      detail: text,
    };
  }
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const useLlmSummary = searchParams.get("use_llm_summary") ?? "true";

  const response = await fetch(
    `${API_BASE_URL}/research/market/india?use_llm_summary=${useLlmSummary}`,
    {
      cache: "no-store",
    }
  );

  const data = await parseBackendResponse(response);

  return NextResponse.json(data, {
    status: response.status,
  });
}