export type HealthResponse = {
  status: string;
  service: string;
  version: string;
};

const API_BASE_URL =
  process.env.INTERNAL_API_BASE_URL ?? "http://localhost:8000/api/v1";

export async function getBackendHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`, {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`Backend health check failed with status ${response.status}`);
  }

  return response.json();
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
  const response = await fetch(`${API_BASE_URL}/market-data/providers/health`, {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(
      `Provider health check failed with status ${response.status}`,
    );
  }

  return response.json();
}