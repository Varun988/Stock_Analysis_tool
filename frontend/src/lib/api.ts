import {
  getBackendHeaders,
  INTERNAL_API_BASE_URL,
  parseBackendResponse,
} from "@/lib/server-api";

export type HealthResponse = {
  status: string;
  service: string;
  version: string;
};

export async function getBackendHealth(): Promise<HealthResponse> {
  const response = await fetch(`${INTERNAL_API_BASE_URL}/health`, {
    method: "GET",
    headers: getBackendHeaders(),
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`Backend health check failed with status ${response.status}`);
  }

  return parseBackendResponse(response) as Promise<HealthResponse>;
}

export type ProviderHealth = {
  configured: boolean;
  status: string;
  description: string;
};

export type ProviderHealthResponse = {
  success: boolean;
  message: string;
  data: Record<string, ProviderHealth>;
};

export async function getProviderHealth(): Promise<ProviderHealthResponse> {
  const response = await fetch(
    `${INTERNAL_API_BASE_URL}/market-data/providers/health`,
    {
      method: "GET",
      headers: getBackendHeaders(),
      cache: "no-store",
    }
  );

  if (!response.ok) {
    throw new Error(
      `Provider health check failed with status ${response.status}`,
    );
  }

  return parseBackendResponse(response) as Promise<ProviderHealthResponse>;
}

export type AIProviderInfo = {
  configured: boolean;
  status: string;
  description: string;
  model?: string;
};

export type AIProviderStatusResponse = {
  success: boolean;
  message: string;
  data: {
    configured_provider: string;
    providers: Record<string, AIProviderInfo>;
  };
};

export async function getAIProviderStatus(): Promise<AIProviderStatusResponse> {
  const response = await fetch(`${INTERNAL_API_BASE_URL}/ai/providers/status`, {
    method: "GET",
    headers: getBackendHeaders(),
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(
      `AI provider status check failed with status ${response.status}`,
    );
  }

  return parseBackendResponse(response) as Promise<AIProviderStatusResponse>;
}
