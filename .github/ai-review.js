import * as core from '@actions/core';
import { execSync } from 'child_process';
import OpenAI from 'openai';

// 1. Collect diff between HEAD and the previous commit on this branch
const before = process.env.GITHUB_EVENT_BEFORE;
const after  = process.env.GITHUB_EVENT_AFTER;
const fullDiff = execSync(`git diff ${before} ${after}`, { encoding: 'utf8' });

// 2. Trim if it exceeds ~10 000 tokens (simple heuristic)
const MAX_CHARS = 30_000;
const diff = fullDiff.length > MAX_CHARS
  ? fullDiff.slice(0, MAX_CHARS) + '\n\n--- DIFF TRUNCATED ---'
  : fullDiff;

// 3. Build the prompt
const system = `
You are an infallible senior engineer.
Given a git diff, decide only:

PASS – if all changes are safe, logically correct, no obvious bugs/security issues.
FAIL – if any change is suspicious or needs human review.

Answer with exactly "PASS" or "FAIL" (no commentary).
`;
const user = `Project context:
<insert any static project docs the model should always see>

Diff under review:
\`\`\`diff
${diff}
\`\`\`
`;

// 4. Call the model
const openai = new OpenAI();
const resp = await openai.chat.completions.create({
  model: 'gpt-4.1-mini',
  temperature: 0,
  messages: [{ role: 'system', content: system },
             { role: 'user',   content: user   }],
});

const verdict = resp.choices[0].message.content.trim().toUpperCase();
core.notice(`Model verdict: ${verdict}`);

if (verdict !== 'PASS') {
  core.setFailed('AI review failed.');
  process.exit(1);
}

