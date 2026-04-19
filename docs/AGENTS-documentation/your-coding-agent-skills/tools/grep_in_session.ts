import { tool } from "@opencode-ai/plugin"
import { createHash } from "node:crypto"
import os from "node:os"
import path from "node:path"

type ExportedMessage = any
type SearchMode = "content" | "tools" | "ops" | "raw"
type SortMode = "recent" | "relevance"

type RenderBlock = {
	id: string
	index: number
	role: string
	ts: string
	text: string
}

type CachedRender = {
	version: number
	sessionID: string
	exportHash: string
	generatedAt: number
	modes: Record<SearchMode, string>
}

const CACHE_VERSION = 2

const OMITTED_KEYS = new Set([
	"reasoningencryptedcontent",
	"itemid",
	"sessionid",
	"messageid",
	"snapshot",
	"tokens",
	"cost",
	"cache",
	"metadata",
	"openai",
	"time",
])

const REDACTION_PATTERNS: Array<{ re: RegExp; replacement: string }> = [
	{ re: /\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b/g, replacement: "[redacted-jwt]" },
	{ re: /\bsk-[A-Za-z0-9]{20,}\b/g, replacement: "[redacted-openai-key]" },
	{ re: /\bAKIA[0-9A-Z]{16}\b/g, replacement: "[redacted-aws-key]" },
	{ re: /\bghp_[A-Za-z0-9]{20,}\b/g, replacement: "[redacted-github-token]" },
	{ re: /\bAIza[0-9A-Za-z_-]{20,}\b/g, replacement: "[redacted-google-key]" },
	{ re: /\bxox[baprs]-[A-Za-z0-9-]{10,}\b/g, replacement: "[redacted-slack-token]" },
	{ re: /\b[a-fA-F0-9]{40,}\b/g, replacement: "[redacted-long-hex]" },
	{ re: /(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{120,}={0,2}(?![A-Za-z0-9+/=])/g, replacement: "[redacted-base64-blob]" },
]

function escapeRegexLiteral(s: string) {
	return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
}

function clip(s: string, maxChars = 220) {
	if (s.length <= maxChars) return s
	return s.slice(0, Math.max(0, maxChars - 3)) + "..."
}

function normalizeWhitespace(s: string) {
	return s.replace(/\s+/g, " ").trim()
}

function looksLikeMessage(value: any) {
	if (!value || typeof value !== "object") return false
	return Boolean(
		value.role ||
		value.author ||
		value.info ||
		value.parts ||
		value.content ||
		value.text ||
		value.message
	)
}

function toPlainText(value: any): string {
	if (value == null) return ""
	if (typeof value === "string") return value
	if (typeof value === "number" || typeof value === "boolean") return String(value)

	if (Array.isArray(value)) {
		return value.map(toPlainText).filter(Boolean).join("\n")
	}

	if (typeof value === "object") {
		if (typeof value.text === "string") return value.text
		if (typeof value.content === "string") return value.content
		if (typeof value.message === "string") return value.message
		if (value.message) return toPlainText(value.message)
	}

	return ""
}

function extractMessageArray(exportJson: any): ExportedMessage[] {
	const directCandidates = [
		exportJson,
		exportJson?.data,
		exportJson?.messages,
		exportJson?.session?.messages,
		exportJson?.result?.messages,
		exportJson?.chat?.messages,
	]

	for (const candidate of directCandidates) {
		if (Array.isArray(candidate)) return candidate
	}

	const queue: any[] = [exportJson]
	const seen = new Set<any>()

	while (queue.length) {
		const current = queue.shift()
		if (!current || typeof current !== "object" || seen.has(current)) continue
		seen.add(current)

		for (const value of Object.values(current)) {
			if (Array.isArray(value) && value.some(looksLikeMessage)) return value as ExportedMessage[]
			if (value && typeof value === "object") queue.push(value)
		}
	}

	return []
}

function normalizeTimestamp(value: any): string {
	if (value == null) return ""
	if (typeof value === "string" || typeof value === "number") return String(value)
	if (typeof value === "object") {
		const candidate = value.created ?? value.updated ?? value.start ?? value.end
		if (candidate == null) return ""
		if (typeof candidate === "string" || typeof candidate === "number") return String(candidate)
	}
	return ""
}

export function redactSensitive(text: string) {
	let out = text
	for (const { re, replacement } of REDACTION_PATTERNS) {
		out = out.replace(re, replacement)
	}
	return out
}

function sanitizeLargeString(value: string, keyName = "value") {
	if (value.length <= 800) return redactSensitive(value)
	const looksDense = !/\s/.test(value.slice(0, 200))
	if (!looksDense) return redactSensitive(value)
	return `[omitted long ${keyName} (${value.length} chars)]`
}

export function sanitizeForOutput(value: any, keyName = ""): any {
	if (value == null) return value

	if (typeof value === "string") return sanitizeLargeString(value, keyName || "value")
	if (typeof value === "number" || typeof value === "boolean") return value

	if (Array.isArray(value)) {
		const cleaned = value
			.map((item) => sanitizeForOutput(item, keyName))
			.filter((item) => item !== undefined)
		return cleaned
	}

	if (typeof value === "object") {
		const out: Record<string, any> = {}
		for (const [k, v] of Object.entries(value)) {
			if (OMITTED_KEYS.has(k.toLowerCase())) continue
			const cleaned = sanitizeForOutput(v, k)
			if (cleaned === undefined) continue
			if (cleaned && typeof cleaned === "object" && !Array.isArray(cleaned) && Object.keys(cleaned).length === 0) continue
			out[k] = cleaned
		}
		return Object.keys(out).length ? out : undefined
	}

	return value
}

function uniqueNonEmpty(values: string[]) {
	const seen = new Set<string>()
	const out: string[] = []
	for (const raw of values) {
		const value = raw.trim()
		if (!value) continue
		const sig = normalizeWhitespace(value)
		if (seen.has(sig)) continue
		seen.add(sig)
		out.push(value)
	}
	return out
}

function formatToolPart(name: string, input: any, output: any) {
	const payload = {
		input: sanitizeForOutput(input),
		output: sanitizeForOutput(output),
	}
	return `**Tool:** \`${name}\`\n\n\`\`\`json\n${JSON.stringify(payload, null, 2)}\n\`\`\``
}

function summarizeToolInput(input: any) {
	const cleaned = sanitizeForOutput(input)
	if (!cleaned || typeof cleaned !== "object" || Array.isArray(cleaned)) return ""

	const c = cleaned as Record<string, any>
	const parts: string[] = []

	if (typeof c.description === "string") parts.push(c.description)
	if (typeof c.command === "string") parts.push(c.command)
	if (typeof c.filePath === "string") parts.push(c.filePath)
	if (typeof c.pattern === "string") parts.push(`pattern=${c.pattern}`)
	if (typeof c.query === "string") parts.push(`query=${c.query}`)
	if (typeof c.url === "string") parts.push(c.url)
	if (typeof c.path === "string") parts.push(`path=${c.path}`)
	if (typeof c.workdir === "string") parts.push(`workdir=${c.workdir}`)

	if (!parts.length) {
		const keys = Object.keys(c).slice(0, 3)
		if (!keys.length) return ""
		return `keys=${keys.join(",")}`
	}

	return clip(parts.join(" | "))
}

function summarizeToolPart(name: string, input: any, output: any) {
	const details = summarizeToolInput(input)
	const hasOutput = output != null
	return `[${name}]${details ? ` ${details}` : ""}${hasOutput ? " | output:yes" : ""}`
}

function partsToMarkdown(parts: any[] | undefined) {
	if (!parts?.length) return ""
	const out: string[] = []

	for (const p of parts) {
		if (!p) continue

		if (p.type === "text" && typeof p.text === "string") {
			out.push(redactSensitive(p.text))
			continue
		}

		if (p.type === "tool") {
			const name = String(p.name ?? p.tool ?? "tool")
			const input = p.input ?? p.args ?? p.arguments ?? p.state?.input
			const output = p.output ?? p.result ?? p.state?.output
			out.push(formatToolPart(name, input, output))
			continue
		}

		out.push("```json\n" + JSON.stringify(sanitizeForOutput(p) ?? {}, null, 2) + "\n```")
	}

	return out.join("\n\n").trim()
}

export function postFilterMarkdown(md: string) {
	const noisyLinePatterns = [
		/reasoningEncryptedContent/i,
		/"itemId"\s*:/i,
		/"sessionID"\s*:/i,
		/"messageID"\s*:/i,
		/"snapshot"\s*:/i,
	]

	const filtered = md
		.split("\n")
		.filter((line) => !noisyLinePatterns.some((re) => re.test(line)))
		.map((line) => line.replace(/[A-Za-z0-9+/_=-]{400,}/g, "[omitted long blob]"))
		.join("\n")

	return redactSensitive(filtered)
}

type SanitizationFixture = {
	name: string
	render: () => string
	absent: string[]
	present?: string[]
}

const SANITIZATION_FIXTURES: SanitizationFixture[] = [
	{
		name: "removes reasoningEncryptedContent lines",
		render: () =>
			postFilterMarkdown(
				"\"itemId\": \"rs_123\"\n\"reasoningEncryptedContent\": \"abc\"\n\"ok\": true"
			),
		absent: ["reasoningEncryptedContent", "itemId"],
		present: ["\"ok\": true"],
	},
	{
		name: "redacts JWT-like tokens",
		render: () => redactSensitive("token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc.def"),
		absent: ["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc.def"],
		present: ["[redacted-jwt]"],
	},
	{
		name: "redacts dense base64 blobs",
		render: () =>
			redactSensitive(
				"blob " + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".repeat(3)
			),
		absent: ["ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"],
		present: ["[redacted-base64-blob]"],
	},
	{
		name: "omits metadata keys during sanitize",
		render: () =>
			JSON.stringify(
				sanitizeForOutput({
					metadata: { openai: { reasoningEncryptedContent: "secret" } },
					sessionID: "abc",
					messageID: "def",
					value: "ok",
				}),
				null,
				2
			),
		absent: ["metadata", "reasoningEncryptedContent", "sessionID", "messageID"],
		present: ["\"value\": \"ok\""],
	},
]

export function runSanitizationFixtureSuite() {
	const results = SANITIZATION_FIXTURES.map((fixture) => {
		const out = fixture.render()
		const missingPresent = (fixture.present ?? []).filter((needle) => !out.includes(needle))
		const leakedAbsent = fixture.absent.filter((needle) => out.includes(needle))
		const passed = missingPresent.length === 0 && leakedAbsent.length === 0
		return {
			name: fixture.name,
			passed,
			missingPresent,
			leakedAbsent,
		}
	})

	return {
		total: results.length,
		passed: results.filter((r) => r.passed).length,
		results,
	}
}

function renderMessageBlocks(exportJson: any, mode: SearchMode): RenderBlock[] {
	const items = extractMessageArray(exportJson)
	const blocks: RenderBlock[] = []

	for (let idx = 0; idx < items.length; idx++) {
		const item = items[idx]
		const info = item?.info ?? item?.message ?? item ?? {}
		const role = String(info.role ?? info.author ?? "unknown")
		const ts = normalizeTimestamp(info.createdAt ?? info.created_at ?? info.time ?? item?.time)

		const parts = item?.parts ?? info.parts ?? item?.message?.parts ?? []

		const textChunks: string[] = []
		const toolChunks: string[] = []
		const opsChunks: string[] = []

		for (const p of parts) {
			if (!p) continue
			if (p.type === "text" && typeof p.text === "string") {
				textChunks.push(redactSensitive(p.text))
				continue
			}

			if (p.type === "tool") {
				const name = String(p.name ?? p.tool ?? "tool")
				const input = p.input ?? p.args ?? p.arguments ?? p.state?.input
				const output = p.output ?? p.result ?? p.state?.output
				toolChunks.push(formatToolPart(name, input, output))
				opsChunks.push(summarizeToolPart(name, input, output))
			}
		}

		const bodyFromContent = redactSensitive(
			toPlainText(
				info.content ??
				item?.content ??
				info.text ??
				item?.text ??
				item?.message?.content ??
				item?.message?.text
			)
		)
		if (bodyFromContent) textChunks.push(bodyFromContent)

		const contentBody = uniqueNonEmpty(textChunks).join("\n\n")
		const toolsBody = uniqueNonEmpty(toolChunks).join("\n\n")
		const opsBody = uniqueNonEmpty(opsChunks).join("\n")
		const rawBody = partsToMarkdown(parts) || contentBody

		const body =
			mode === "content"
				? contentBody
				: mode === "tools"
					? toolsBody
					: mode === "ops"
						? opsBody
						: rawBody

		if (!body.trim()) continue

		const blockText = postFilterMarkdown(`## ${role}${ts ? ` — ${ts}` : ""}\n\n${body.trim()}\n`)
		if (!blockText.trim()) continue

		blocks.push({
			id: `${idx + 1}`,
			index: idx,
			role,
			ts,
			text: blockText.trimEnd() + "\n",
		})
	}

	return blocks
}

function blocksToMarkdown(blocks: RenderBlock[]) {
	const rendered = blocks.map((b) => b.text.trim()).filter(Boolean).join("\n\n").trim()
	if (!rendered) return ""
	return rendered + "\n"
}

function markdownToBlocks(md: string): RenderBlock[] {
	const trimmed = md.trim()
	if (!trimmed) return []

	const chunks = trimmed.startsWith("## ") ? trimmed.split(/\n(?=##\s)/g) : [trimmed]

	return chunks
		.map((chunk, idx) => ({
			id: `${idx + 1}`,
			index: idx,
			role: "unknown",
			ts: "",
			text: chunk.trimEnd() + "\n",
		}))
		.filter((b) => b.text.trim())
}

function buildModeMarkdownMap(exportJson: any): Record<SearchMode, string> {
	const content = blocksToMarkdown(renderMessageBlocks(exportJson, "content"))
	const tools = blocksToMarkdown(renderMessageBlocks(exportJson, "tools"))
	const ops = blocksToMarkdown(renderMessageBlocks(exportJson, "ops"))
	const raw = blocksToMarkdown(renderMessageBlocks(exportJson, "raw"))

	const fallbackRaw =
		raw ||
		"```json\n" +
		JSON.stringify(sanitizeForOutput(exportJson) ?? {}, null, 2) +
		"\n```\n"

	return {
		content: postFilterMarkdown(content),
		tools: postFilterMarkdown(tools),
		ops: postFilterMarkdown(ops),
		raw: postFilterMarkdown(fallbackRaw),
	}
}

function countMatches(text: string, re: RegExp) {
	const flags = re.flags.includes("g") ? re.flags : re.flags + "g"
	const probe = new RegExp(re.source, flags)
	let count = 0
	let match: RegExpExecArray | null = null

	while ((match = probe.exec(text))) {
		count += 1
		if (match[0].length === 0) probe.lastIndex += 1
		if (count >= 5000) break
	}

	return count
}

function signatureForBlock(text: string) {
	const normalized = text
		.toLowerCase()
		.replace(/\d{5,}/g, "#")
		.replace(/\s+/g, " ")
		.trim()

	return createHash("sha1").update(normalized).digest("hex")
}

function findBlockMatches(
	blocks: RenderBlock[],
	re: RegExp,
	sort: SortMode,
	maxMatches: number
) {
	const found: Array<{ block: RenderBlock; score: number }> = []

	for (const block of blocks) {
		re.lastIndex = 0
		if (!re.test(block.text)) continue
		const score = Math.max(1, countMatches(block.text, re))
		found.push({ block, score })
	}

	if (sort === "relevance") {
		found.sort((a, b) => b.score - a.score || b.block.index - a.block.index)
	} else {
		found.sort((a, b) => b.block.index - a.block.index)
	}

	const deduped: Array<{ block: RenderBlock; score: number }> = []
	const seen = new Set<string>()

	for (const entry of found) {
		const signature = signatureForBlock(entry.block.text)
		if (seen.has(signature)) continue
		seen.add(signature)
		deduped.push(entry)
		if (deduped.length >= maxMatches) break
	}

	return deduped
}

function renderBlockMatches(matches: Array<{ block: RenderBlock; score: number }>) {
	return matches
		.map((entry, idx) => {
			const label = `#${idx + 1} (message ${entry.block.index + 1}, score ${entry.score})`
			return `${label}\n\`\`\`md\n${entry.block.text.trimEnd()}\n\`\`\``
		})
		.join("\n\n")
}

export function applyOutputBudget(text: string, maxChars: number) {
	if (text.length <= maxChars) return text
	const suffix = `\n\n...[truncated at ${maxChars} chars; narrow query or reduce maxMatches]`
	const allowed = Math.max(0, maxChars - suffix.length)
	return text.slice(0, allowed) + suffix
}

function cacheFilePath(sessionID: string, exportHash: string) {
	return path.join(os.tmpdir(), `opencode-session-cache-${sessionID}-${exportHash}.json`)
}

async function readCachedRender(cachePath: string): Promise<CachedRender | null> {
	try {
		const file = Bun.file(cachePath)
		if (!(await file.exists())) return null
		const parsed = await file.json()
		if (!parsed || typeof parsed !== "object") return null
		if (parsed.version !== CACHE_VERSION) return null
		if (!parsed.modes || typeof parsed.modes !== "object") return null
		return parsed as CachedRender
	} catch {
		return null
	}
}

async function writeCachedRender(cachePath: string, payload: CachedRender) {
	try {
		await Bun.write(cachePath, JSON.stringify(payload))
	} catch {
		// Best-effort cache only.
	}
}

export default tool({
	description:
		"Export the current OpenCode session to a filtered /export-like transcript in memory, then search it with block-aware results, dedupe, sorting, and output budget controls. Prefer ops/content/tools modes; use raw as a last resort.",
	args: {
		query: tool.schema.string().describe("Keyword or regex to search for"),
		regex: tool.schema.boolean().default(false).describe("Treat query as regex (default: literal keyword search)"),
		flags: tool.schema.string().default("i").describe("Regex flags (e.g. i, m, s)"),
		contextLines: tool.schema.number().int().min(0).max(20).default(2).describe("Context lines for rg engine output"),
		maxMatches: tool.schema.number().int().min(1).max(50).default(10).describe("Maximum number of match blocks to return"),
		maxOutputChars: tool.schema.number().int().min(500).max(120000).default(12000).describe("Hard cap for final response size"),
		mode: tool.schema
			.enum(["content", "tools", "ops", "raw"])
			.default("ops")
			.describe("Transcript mode: content, tools, ops, or raw (raw is a last resort)"),
		sort: tool.schema
			.enum(["recent", "relevance"])
			.default("recent")
			.describe("Match sorting: recent first or relevance first"),
		debug: tool.schema.boolean().default(false).describe("Include diagnostics in output"),
		selfTest: tool.schema.boolean().default(false).describe("Run built-in sanitization fixtures and return pass/fail summary"),
		engine: tool.schema
			.enum(["js", "rg"])
			.default("js")
			.describe("Search engine: js (block-aware) or rg (line-oriented)"),
	},
	async execute(args, context) {
		if (args.selfTest) {
			const suite = runSanitizationFixtureSuite()
			const lines = [
				`fixtures: ${suite.passed}/${suite.total} passed`,
				...suite.results.map((r) => {
					if (r.passed) return `- PASS ${r.name}`
					const bits: string[] = []
					if (r.missingPresent.length) bits.push(`missing=${r.missingPresent.join(",")}`)
					if (r.leakedAbsent.length) bits.push(`leaked=${r.leakedAbsent.join(",")}`)
					return `- FAIL ${r.name} (${bits.join("; ")})`
				}),
			]
			return applyOutputBudget(lines.join("\n"), args.maxOutputChars)
		}

		const sessionID = context.sessionID
		if (!sessionID) return "No sessionID found in tool context."

		let raw = ""
		try {
			raw = await Bun.$`opencode export ${sessionID}`.text()
		} catch (e: any) {
			return `Failed to run \`opencode export ${sessionID}\`: ${String(e?.message ?? e)}`
		}

		const exportHash = createHash("sha256").update(raw).digest("hex").slice(0, 20)
		const cachePath = cacheFilePath(sessionID, exportHash)

		let cacheHit = false
		let modes: Record<SearchMode, string> | null = null

		const cached = await readCachedRender(cachePath)
		if (cached?.modes) {
			cacheHit = true
			modes = cached.modes
		}

		let exportJson: any | undefined
		if (!modes) {
			try {
				exportJson = JSON.parse(raw)
			} catch {
				return `opencode export did not return valid JSON. First 300 chars:\n${raw.slice(0, 300)}`
			}

			modes = buildModeMarkdownMap(exportJson)
			await writeCachedRender(cachePath, {
				version: CACHE_VERSION,
				sessionID,
				exportHash,
				generatedAt: Date.now(),
				modes,
			})
		}

		const mode = args.mode as SearchMode
		const sort = args.sort as SortMode
		const md = modes[mode] ?? ""

		if (args.debug) {
			if (!exportJson) {
				try {
					exportJson = JSON.parse(raw)
				} catch {
					exportJson = undefined
				}
			}

			const items = exportJson ? extractMessageArray(exportJson) : []
			const debugText = [
				`sessionID: ${sessionID}`,
				`exportHash: ${exportHash}`,
				`cacheHit: ${cacheHit}`,
				`cachePath: ${cachePath}`,
				`mode: ${mode}`,
				`sort: ${sort}`,
				`engine: ${args.engine}`,
				`rawChars: ${raw.length}`,
				`modeChars: ${md.length}`,
				`detectedItems: ${items.length}`,
				"markdownPreview:",
				"```md",
				md.slice(0, 1200),
				"```",
			].join("\n")
			return applyOutputBudget(debugText, args.maxOutputChars)
		}

		if (!md.trim()) return "No searchable content for this mode."

		if (args.engine === "rg") {
			const tmp = path.join(os.tmpdir(), `opencode-session-${sessionID}-${mode}.md`)
			try {
				await Bun.write(tmp, md)

				const rgArgs = [
					"rg",
					"-n",
					"-C",
					String(args.contextLines),
					"--max-count",
					String(args.maxMatches),
				]

				if ((args.flags ?? "").includes("i")) rgArgs.push("-i")

				if (args.regex) {
					rgArgs.push("-e", args.query)
				} else {
					rgArgs.push("-F", "-e", args.query)
				}

				rgArgs.push(tmp)

				const proc = Bun.spawn(rgArgs, {
					stdout: "pipe",
					stderr: "pipe",
				})

				const stdout = await new Response(proc.stdout).text()
				const stderr = await new Response(proc.stderr).text()
				const code = await proc.exited

				if (code === 1) return "No matches."
				if (code !== 0) {
					return `ripgrep search failed (exit ${code}): ${stderr.trim() || "unknown error"}`
				}

				return applyOutputBudget(stdout.trim() || "No matches.", args.maxOutputChars)
			} catch (e: any) {
				return `ripgrep search failed (is rg installed?): ${String(e?.message ?? e)}`
			}
		}

		let re: RegExp
		try {
			const pattern = args.regex ? args.query : escapeRegexLiteral(args.query)
			re = new RegExp(pattern, args.flags)
		} catch (e: any) {
			return `Invalid regex: ${String(e?.message ?? e)}`
		}

		const blocks = markdownToBlocks(md)
		if (!blocks.length) return "No searchable content for this mode."

		const hits = findBlockMatches(blocks, re, sort, args.maxMatches)
		if (!hits.length) return "No matches."

		return applyOutputBudget(renderBlockMatches(hits), args.maxOutputChars)
	},
})
