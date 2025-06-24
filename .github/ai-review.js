import * as core from '@actions/core';
import { execSync } from 'child_process';
import OpenAI from 'openai';
import { z } from 'zod';
import { zodToJsonSchema } from 'zod-to-json-schema';   // npm i zod zod-to-json-schema

(async () => {
  try {
    // 1‒ Collect the diff between the pushed commits
    const before = process.env.GITHUB_EVENT_BEFORE;
    const after  = process.env.GITHUB_EVENT_AFTER;
    const fullDiff = execSync(`git diff ${before} ${after}`, { encoding: 'utf8' });

    // 2‒ Cheap guard against huge payloads (~10 k tokens ≈ 30 k chars)
    const MAX_CHARS = 30_000;
    const diff = fullDiff.length > MAX_CHARS
      ? `${fullDiff.slice(0, MAX_CHARS)}\n\n--- DIFF TRUNCATED ---`
      : fullDiff;

    // 3‒ Single-field schema describing the only answer we accept
    const Verdict = z.object({
      /** "PASS" if everything is OK, otherwise "FAIL" */
      verdict: z.enum(['PASS', 'FAIL'])
    }).strict();

    // 4‒ Fire the request using the Structured-Outputs API
    const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    const resp = await openai.chat.completions.create({
      model: 'gpt-4.1-mini',          // 06-2025 default, lowest latency
      temperature: 0,
      messages: [
        { role: 'system', content: 'You are a meticulous senior engineer. ' +
          'Return a JSON object that satisfies the provided schema ­— nothing else.' },
        { role: 'user', content: `Project context:
This project is a python backend for LLM training, inference and visualization. There is a backend server under server/ that managest the commands and initializes any training or generation. Under ui/ there is a fronend that visualizes the process and the commands for a user to use the architecture from a remote browser. 

The core of the model is written in python using JAX. The model has custom syntethic datadeneration, KV caching at inference, RL workflow and much more in coresponding scripts.

Diff under review:
\`\`\`diff
${diff}
\`\`\`
if the code of this diff seems wrong, inconsistant or wrong syntax, return the json with a FAIL.
` }
      ],
      response_format: {
        type: 'json_schema',
        schema: zodToJsonSchema(Verdict)   // converts Zod → JSON-Schema Draft-2020-12
      }
    });

    // 5‒ Strictly parse the answer and act on it
    const { verdict } = Verdict.parse(JSON.parse(resp.choices[0].message.content));
    core.notice(`Model verdict: ${verdict}`);
    if (verdict !== 'PASS') {
      core.setFailed('AI review failed.');
      process.exit(1);
    }
  } catch (err) {
    core.setFailed(err instanceof Error ? err.message : String(err));
  }
})();

