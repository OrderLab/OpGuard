import { DurableObject } from 'cloudflare:workers';
import { handle, reserve, settle } from './core.js';

export class DemoQuota extends DurableObject {
  async reserve(ip, budget) {
    return this.ctx.storage.transaction(async tx => {
      const result = reserve(await tx.get('entries'), ip, budget, Date.now(), crypto.randomUUID());
      await tx.put('entries', result.entries);
      return { id: result.id, retry: result.retry, reason: result.reason };
    });
  }
  async settle(id, usage) {
    return this.ctx.storage.transaction(async tx => {
      await tx.put('entries', settle(await tx.get('entries') || [], id, usage));
    });
  }
}

export default {
  async fetch(request, env) {
    if (env.DEMO_MODE !== 'live') return handle(request, env);
    // One shared object makes global limits atomic across regions and isolates.
    const quota = env.DEMO_QUOTA.getByName('groq-demo-v1');
    const ip = request.headers.get('CF-Connecting-IP') || 'unknown';
    const hash = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(`${env.GROQ_API_KEY || ''}:${ip}`));
    const visitor = Array.from(new Uint8Array(hash), b => b.toString(16).padStart(2, '0')).join('');
    try {
      return await handle(request, env, { reserve: budget => quota.reserve(visitor, budget), settle: (id, usage) => quota.settle(id, usage) });
    } catch {
      return Response.json({ error: 'The demo is temporarily unavailable.' }, { status: 503, headers: { 'Cache-Control': 'no-store' } });
    }
  },
};
