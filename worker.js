function getCors(request) {
  const origin = request.headers.get('Origin') || '';
  const allowed = ['https://maepixel.pages.dev', 'http://localhost:8790', 'http://127.0.0.1:8790'];
  const allowOrigin = allowed.includes(origin) ? origin : allowed[0];
  return {
    'Access-Control-Allow-Origin': allowOrigin,
    'Access-Control-Allow-Methods': 'GET, POST, DELETE, PATCH, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, X-Admin-Key',
    'Access-Control-Allow-Credentials': 'false',
  };
}

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status, headers: { ...getCors(request), 'Content-Type': 'application/json' },
  });
}
function err(msg, status = 400) { return json({ ok: false, error: msg }, status); }
function pct(a, b) { if (!b) return '0%'; return Math.round((a / b) * 100) + '%'; }

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: getCors(request) });
    }

    // ── POST /track ───────────────────────────────────────────────
    if (request.method === 'POST' && url.pathname === '/track') {
      let body;
      try {
        const text = await request.text();
        body = JSON.parse(text);
      } catch { return err('invalid JSON'); }
      const uid     = (body.uid   || 'unknown').slice(0, 64);
      const event   = (body.event || 'unknown').slice(0, 64);
      const ref     = (body.ref   || '').slice(0, 128);
      const extra   = body.img_type ? body.img_type.slice(0, 64) : (body.slot ? body.slot.slice(0,8) : null);
      const device  = body.deviceType ? body.deviceType.slice(0, 20) : null;
      const os      = body.os ? body.os.slice(0, 20) : null;
      const browser = body.browser ? body.browser.slice(0, 20) : null;
      const screen  = body.screenW ? `${body.screenW}x${body.screenH}` : null;
      const deviceStr = [device, os, browser, screen].filter(Boolean).join('|') || null;
      const country = request.headers.get('CF-IPCountry') || 'unknown';
      const ua = request.headers.get('User-Agent') || '';
      if (/bot|crawl|spider|headless/i.test(ua)) return json({ ok: true, ignored: true });
      await env.DB.prepare(
        `INSERT INTO events (uid, event, ref, country, extra, device) VALUES (?, ?, ?, ?, ?, ?)`
      ).bind(uid, event, ref, country, extra, deviceStr).run();
      return json({ ok: true });
    }

    // ── POST /feedback ────────────────────────────────────────────
    if (request.method === 'POST' && url.pathname === '/feedback') {
      let body; try { body = await request.json(); } catch { return err('invalid JSON'); }
      const rating  = parseInt(body.rating) || null;
      const message = (body.message || '').trim().slice(0, 2000);
      const uid     = (body.uid || 'unknown').slice(0, 64);
      if (!rating && !message) return err('empty submission');
      const ip = request.headers.get('CF-Connecting-IP') || 'unknown';
      const recent = await env.DB.prepare(
        `SELECT COUNT(*) as cnt FROM feedback WHERE ip = ? AND created_at > datetime('now', '-1 hour')`
      ).bind(ip).first();
      if (recent && recent.cnt >= 5) return err('too many submissions', 429);
      await env.DB.prepare(
        `INSERT INTO feedback (rating, message, ip, uid) VALUES (?, ?, ?, ?)`
      ).bind(rating, message || null, ip, uid).run();
      return json({ ok: true });
    }

    // ── GET /stats ────────────────────────────────────────────────
    if (request.method === 'GET' && url.pathname === '/stats') {
      const key = url.searchParams.get('key');
      if (key !== env.ADMIN_KEY) return err('unauthorized', 401);
      const [
        totalEvents, todayEvents, uniqueUsers, todayUnique,
        conversionRaw, topCountries, topRefs,
        eventBreakdown, returning, recentFeedback, imgTypes, deviceBreakdown,
      ] = await Promise.all([
        env.DB.prepare(`SELECT COUNT(*) as n FROM events`).first(),
        env.DB.prepare(`SELECT COUNT(*) as n FROM events WHERE created_at > datetime('now','-1 day')`).first(),
        env.DB.prepare(`SELECT COUNT(DISTINCT uid) as n FROM events`).first(),
        env.DB.prepare(`SELECT COUNT(DISTINCT uid) as n FROM events WHERE created_at > datetime('now','-1 day')`).first(),
        env.DB.prepare(`SELECT
            COUNT(DISTINCT CASE WHEN event='page_open'    THEN uid END) as openers,
            COUNT(DISTINCT CASE WHEN event='export'       THEN uid END) as exporters,
            COUNT(DISTINCT CASE WHEN event='compose'      THEN uid END) as composers,
            COUNT(DISTINCT CASE WHEN event='image_upload' THEN uid END) as uploaders
          FROM events`).first(),
        env.DB.prepare(`SELECT country, COUNT(*) as n FROM events GROUP BY country ORDER BY n DESC LIMIT 8`).all(),
        env.DB.prepare(`SELECT ref, COUNT(DISTINCT uid) as users FROM events WHERE ref!='' GROUP BY ref ORDER BY users DESC LIMIT 8`).all(),
        env.DB.prepare(`SELECT event, COUNT(*) as n FROM events GROUP BY event ORDER BY n DESC`).all(),
        env.DB.prepare(`SELECT COUNT(*) as n FROM (SELECT uid FROM events GROUP BY uid HAVING COUNT(DISTINCT date(created_at))>1)`).first(),
        env.DB.prepare(`SELECT rating, message, created_at FROM feedback ORDER BY created_at DESC LIMIT 20`).all(),
        env.DB.prepare(`SELECT extra as img_type, COUNT(*) as n FROM events WHERE event='image_upload' AND extra IS NOT NULL GROUP BY extra ORDER BY n DESC`).all(),
        env.DB.prepare(`SELECT device, COUNT(*) as n FROM events WHERE device IS NOT NULL GROUP BY device ORDER BY n DESC LIMIT 10`).all(),
      ]);
      const openers  = conversionRaw?.openers  || 1;
      const exporters = conversionRaw?.exporters || 0;
      const composers = conversionRaw?.composers || 0;
      const uploaders = conversionRaw?.uploaders || 0;
      return json({
        ok: true,
        summary: {
          total_events:    totalEvents?.n  || 0,
          events_today:    todayEvents?.n  || 0,
          unique_users:    uniqueUsers?.n  || 0,
          unique_today:    todayUnique?.n  || 0,
          returning_users: returning?.n    || 0,
        },
        funnel: {
          page_opens:   openers,
          uploaded:     uploaders,
          composed:     composers,
          exported:     exporters,
          upload_rate:  pct(uploaders, openers),
          compose_rate: pct(composers, openers),
          export_rate:  pct(exporters, openers),
        },
        top_countries:   topCountries.results,
        top_refs:        topRefs.results,
        event_breakdown: eventBreakdown.results,
        recent_feedback: recentFeedback.results,
        image_types:     imgTypes.results,
        device_breakdown: deviceBreakdown.results,
      });
    }

    // ── GET /feedback ─────────────────────────────────────────────
    if (request.method === 'GET' && url.pathname === '/feedback') {
      const key = url.searchParams.get('key');
      if (key !== env.ADMIN_KEY) return err('unauthorized', 401);
      const rows = await env.DB.prepare(
        `SELECT id, rating, message, created_at FROM feedback ORDER BY created_at DESC LIMIT 200`
      ).all();
      return json({ ok: true, total: rows.results.length, feedback: rows.results });
    }

    // ── GET /showcase — public ────────────────────────────────────
    if (request.method === 'GET' && url.pathname === '/showcase') {
      const rows = await env.DB.prepare(
        `SELECT id, type, img_url, img_a_url, img_b_url, output_url, label, sort_order
         FROM showcase WHERE active=1 ORDER BY sort_order ASC, id DESC LIMIT 30`
      ).all();
      return json({ ok: true, items: rows.results });
    }

    // ── GET /showcase/list — admin ────────────────────────────────
    if (request.method === 'GET' && url.pathname === '/showcase/list') {
      const key = url.searchParams.get('key');
      if (key !== env.ADMIN_KEY) return err('unauthorized', 401);
      const rows = await env.DB.prepare(
        `SELECT * FROM showcase ORDER BY sort_order ASC, id DESC`
      ).all();
      return json({ ok: true, items: rows.results });
    }

    // ── POST /showcase — admin ────────────────────────────────────
    if (request.method === 'POST' && url.pathname === '/showcase') {
      const key = request.headers.get('X-Admin-Key') || url.searchParams.get('key');
      if (key !== env.ADMIN_KEY) return err('unauthorized', 401);
      let body; try { body = await request.json(); } catch { return err('invalid JSON'); }
      const type       = (body.type       || 'hero').slice(0, 20);
      const img_url    = (body.img_url    || '').slice(0, 500);
      const img_a_url  = (body.img_a_url  || '').slice(0, 500);
      const img_b_url  = (body.img_b_url  || '').slice(0, 500);
      const output_url = (body.output_url || '').slice(0, 500);
      const label      = (body.label      || '').slice(0, 100);
      const sort_order = parseInt(body.sort_order) || 0;
      const result = await env.DB.prepare(
        `INSERT INTO showcase (type,img_url,img_a_url,img_b_url,output_url,label,sort_order,active)
         VALUES (?,?,?,?,?,?,?,1)`
      ).bind(type,img_url,img_a_url,img_b_url,output_url,label,sort_order).run();
      return json({ ok: true, id: result.meta?.last_row_id });
    }

    // ── DELETE /showcase/:id — admin ──────────────────────────────
    if (request.method === 'DELETE' && url.pathname.startsWith('/showcase/')) {
      const key = request.headers.get('X-Admin-Key') || url.searchParams.get('key');
      if (key !== env.ADMIN_KEY) return err('unauthorized', 401);
      const id = parseInt(url.pathname.split('/').pop());
      if (!id) return err('invalid id');
      await env.DB.prepare(`UPDATE showcase SET active=0 WHERE id=?`).bind(id).run();
      return json({ ok: true, id });
    }

    // ── PATCH /showcase/:id — admin ───────────────────────────────
    if (request.method === 'PATCH' && url.pathname.startsWith('/showcase/')) {
      const key = request.headers.get('X-Admin-Key') || url.searchParams.get('key');
      if (key !== env.ADMIN_KEY) return err('unauthorized', 401);
      const id = parseInt(url.pathname.split('/').pop());
      let body; try { body = await request.json(); } catch { return err('invalid JSON'); }
      const fields = []; const vals = [];
      if (body.label      !== undefined){ fields.push('label=?');      vals.push(String(body.label).slice(0,100)); }
      if (body.sort_order !== undefined){ fields.push('sort_order=?'); vals.push(parseInt(body.sort_order)||0); }
      if (body.active     !== undefined){ fields.push('active=?');     vals.push(body.active?1:0); }
      if (!fields.length) return err('nothing to update');
      vals.push(id);
      await env.DB.prepare(`UPDATE showcase SET ${fields.join(',')} WHERE id=?`).bind(...vals).run();
      return json({ ok: true, id });
    }

    return err('not found', 404);
  },
};
