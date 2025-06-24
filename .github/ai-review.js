import * as core from '@actions/core';
import { execSync } from 'child_process';
import OpenAI from 'openai';
import { z } from 'zod';
import { zodToJsonSchema } from 'zod-to-json-schema';

(async () => {
  try {
    /* ---------------- collect the diff ---------------- */
    const before = process.env.GITHUB_EVENT_BEFORE;
    const after  = process.env.GITHUB_EVENT_AFTER;
    const fullDiff = execSync(`git diff ${before} ${after}`, { encoding: 'utf8' });

    const MAX_CHARS = 30_000;                               // ~10 k tokens
    const diff = fullDiff.length > MAX_CHARS
      ? `${fullDiff.slice(0, MAX_CHARS)}\n\n--- DIFF TRUNCATED ---`
      : fullDiff;

    console.log('----DIFF START----');
    console.log(diff);
    console.log('----DIFF END----');

    /* ---------------- define the allowed reply shape --- */
    const Verdict = z.object({
      verdict: z.enum(['PASS', 'FAIL'])
    }).strict();

    // Build a “function-style” schema: name + schema
    const jsonSchema = {
      name: 'ai_review_verdict',
      description: 'PASS if diff looks safe, otherwise FAIL.',
      schema: zodToJsonSchema(Verdict)          // type/object/properties/required
    };

    /* ---------------- call OpenAI ---------------------- */
    const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

    const resp = await openai.chat.completions.create({
      model: 'gpt-4.1-mini',
      temperature: 0,
      messages: [
        {
          role: 'system',
          content:
            'You are a meticulous senior engineer. ' +
            'Return a JSON object that satisfies the provided schema — nothing else.'
        },
        {
          role: 'user',
          content: `Project context:
This project is a python backend for LLM training, inference and visualization. There is a backend server under server/ that managest the commands and initializes any training or generation. Under ui/ there is a fronend that visualizes the process and the commands for a user to use the architecture from a remote browser. 

The core of the model is written in python using JAX. The model has custom syntethic datadeneration, KV caching at inference, RL workflow and much more in coresponding scripts.

Diff under review:
\`\`\`diff
${diff}
\`\`\`
If the code of this diff seems wrong, inconsistent or wrong syntax, return the JSON with a FAIL.`
        }
      ],
      response_format: {
        type: 'json_schema',
        json_schema: jsonSchema                       // <-- correct key & structure
      }
    });

    /* ---------------- inspect + act on reply ----------- */
    console.log('----RESPONSE START----');
    console.log(resp.choices[0].message.content);
    console.log('----RESPONSE END----');

    const { verdict } = Verdict.parse(
      JSON.parse(resp.choices[0].message.content)
    );

    core.notice(`Model verdict: ${verdict}`);
    console.log(`Model verdict: ${verdict}`);

    if (verdict !== 'PASS') {
      core.setFailed('AI review failed.');
      process.exit(1);
    }
  } catch (err) {
    core.setFailed(err instanceof Error ? err.message : String(err));
  }
})();
