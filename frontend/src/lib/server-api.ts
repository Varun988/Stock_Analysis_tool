export const INTERNAL_API_BASE_URL =
  process.env.INTERNAL_API_BASE_URL ?? "http://localhost:8000/api/v1";

export const INTERNAL_API_KEY_HEADER = "X-Internal-API-Key";

export function getBackendHeaders(extraHeaders?: HeadersInit): Headers {
  const headers = new Headers(extraHeaders);
  const internalApiKey = process.env.INTERNAL_API_KEY;

  if (internalApiKey) {
    headers.set(INTERNAL_API_KEY_HEADER, internalApiKey);
  }

  return headers;
}

export async function parseBackendResponse(response: Response) {
  const responseText = await response.text();

  if (!responseText) {
    return {
      detail: "Backend returned an empty response.",
    };
  }

  try {
    return JSON.parse(responseText);
  } catch {
    return {
      detail: responseText,
    };
  }
}
