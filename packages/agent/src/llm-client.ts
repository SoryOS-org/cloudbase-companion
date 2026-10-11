/**
 * @soryos/agent
 * Real streaming LLM client. Resolves the API key through @soryos/credentials
 * (session override → settings store → env var) and calls the provider API.
 */

import { credentialManager } from "@soryos/credentials";
import { SUPPORTED_AI_MODELS } from "@soryos/provider";

export interface ChatMessageInput {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface StreamChatOptions {
  providerId: string;
  model: string;
  messages: ChatMessageInput[];
  sessionId?: string;
  signal?: AbortSignal;
  onDelta: (text: string) => void;
}

const OPENAI_COMPATIBLE_ENDPOINTS: Record<string, string> = {
  openai: "https://api.openai.com/v1",
  google: "https://generativelanguage.googleapis.com/v1beta/openai",
  mistral: "https://api.mistral.ai/v1",
  openrouter: "https://openrouter.ai/api/v1",
  deepseek: "https://api.deepseek.com/v1",
  grok: "https://api.x.ai/v1",
  "opencode-zen": "https://zen.opencode.ai/api/v1",
};

const PROVIDER_ALIASES: Record<string, string> = {
  gemini: "google",
  "google gemini": "google",
  xai: "grok",
  zen: "opencode-zen",
  opencode: "opencode-zen",
  claude: "anthropic",
};

/** Normalizes a provider label ("DeepSeek", "Gemini") to its registry id. */
export function normalizeProviderId(raw?: string | null): string | undefined {
  if (!raw) return undefined;
  const id = raw.trim().toLowerCase();
  return PROVIDER_ALIASES[id] || id;
}

/** Finds the provider owning a model id, if known. */
export function providerForModel(model?: string | null): string | undefined {
  if (!model) return undefined;
  return SUPPORTED_AI_MODELS.find((m) => m.id === model)?.providerId;
}

async function readSSE(
  response: Response,
  onData: (data: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  if (!response.body) throw new Error("Réponse vide du fournisseur IA");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    if (signal?.aborted) break;
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith("data:")) onData(trimmed.slice(5).trim());
    }
  }
  const rest = buffer.trim();
  if (rest.startsWith("data:")) onData(rest.slice(5).trim());
}

export async function streamChat(options: StreamChatOptions): Promise<string> {
  const providerId = normalizeProviderId(options.providerId) || "opencode-zen";
  const cred = credentialManager.resolveCredentials(providerId, options.sessionId);
  if (!cred.isConfigured || !cred.apiKey) {
    throw new Error(
      `Aucune clé API configurée pour « ${providerId} ». Ajoutez-la dans Paramètres → Fournisseurs IA.`,
    );
  }

  let full = "";
  const emit = (text: string) => {
    if (!text) return;
    full += text;
    options.onDelta(text);
  };

  if (providerId === "anthropic") {
    const system = options.messages.filter((m) => m.role === "system").map((m) => m.content).join("\n\n");
    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      signal: options.signal,
      headers: {
        "Content-Type": "application/json",
        "x-api-key": cred.apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: options.model,
        max_tokens: 8192,
        stream: true,
        ...(system ? { system } : {}),
        messages: options.messages.filter((m) => m.role !== "system"),
      }),
    });
    if (!response.ok) {
      throw new Error(`Erreur Anthropic ${response.status} : ${(await response.text()).slice(0, 500)}`);
    }
    await readSSE(response, (data) => {
      try {
        const parsed = JSON.parse(data);
        if (parsed.type === "content_block_delta" && parsed.delta?.type === "text_delta") emit(parsed.delta.text);
        if (parsed.type === "error") throw new Error(parsed.error?.message || "Erreur Anthropic");
      } catch (e) {
        if (e instanceof Error && !(e instanceof SyntaxError)) throw e;
      }
    }, options.signal);
    return full;
  }

  const endpoint = OPENAI_COMPATIBLE_ENDPOINTS[providerId];
  if (!endpoint) throw new Error(`Fournisseur IA inconnu : ${providerId}`);

  const response = await fetch(`${endpoint}/chat/completions`, {
    method: "POST",
    signal: options.signal,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${cred.apiKey}`,
    },
    body: JSON.stringify({ model: options.model, messages: options.messages, stream: true }),
  });
  if (!response.ok) {
    throw new Error(`Erreur ${providerId} ${response.status} : ${(await response.text()).slice(0, 500)}`);
  }
  await readSSE(response, (data) => {
    if (data === "[DONE]") return;
    try {
      const parsed = JSON.parse(data);
      if (parsed.error) throw new Error(parsed.error.message || JSON.stringify(parsed.error));
      emit(parsed.choices?.[0]?.delta?.content || "");
    } catch (e) {
      if (e instanceof Error && !(e instanceof SyntaxError)) throw e;
    }
  }, options.signal);
  return full;
}
