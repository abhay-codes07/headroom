import type { Plugin } from "@opencode-ai/plugin";
import { tool } from "@opencode-ai/plugin";
import { z } from "zod";

import { createHeadroomRetrieveTool, getDefaultProxyUrl } from "./retrieve.js";
import { installHeadroomTransport } from "./transport.js";

export interface HeadroomOpenCodePluginOptions {
  proxyUrl?: string;
  project?: string;
  backend?: string;
  debug?: boolean;
}

function normalizeProxyUrl(url: string): string {
  return url.replace(/\/+$/, "");
}

function resolveProxyUrl(options?: HeadroomOpenCodePluginOptions): string {
  return normalizeProxyUrl(
    options?.proxyUrl ??
      process.env.HEADROOM_PROXY_URL ??
      process.env.HEADROOM_BASE_URL ??
      getDefaultProxyUrl(),
  );
}

function resolveProject(
  options: HeadroomOpenCodePluginOptions,
  input: { project?: unknown; directory?: string },
): string | undefined {
  return (
    options.project ??
    (input.project as { id?: string } | undefined)?.id ??
    input.directory
  );
}

export const HeadroomPlugin: Plugin = async (input, options = {}) => {
  const pluginOptions = options as HeadroomOpenCodePluginOptions;
  const proxyUrl = resolveProxyUrl(pluginOptions);
  // Resolve the project once so the transport header and the spawned-subprocess
  // env var agree, and per-project savings attribution works for the proxied
  // HTTP requests too, not just child processes (#2847).
  const project = resolveProject(pluginOptions, input);
  const retrieveTool = createHeadroomRetrieveTool({ proxyBaseUrl: proxyUrl });
  const uninstallTransport = installHeadroomTransport({
    proxyUrl,
    debug: pluginOptions.debug,
    project,
  });

  return {
    dispose: async () => {
      uninstallTransport();
    },
    tool: {
      headroom_retrieve: tool({
        description: retrieveTool.description,
        args: {
          hash: z
            .string()
            .regex(/^[a-f0-9]{24}$/i, "Expected 24-character hex hash"),
        },
        async execute(args) {
          return retrieveTool.execute(args);
        },
      }),
    },
    "shell.env": async (_input, output) => {
      output.env.HEADROOM_ACTIVE = "1";
      output.env.HEADROOM_PROXY_URL = proxyUrl;
      if (project) {
        output.env.HEADROOM_PROJECT = project;
      }
      if (pluginOptions.backend) {
        output.env.HEADROOM_BACKEND = pluginOptions.backend;
      }
    },
  };
};

export default HeadroomPlugin;
