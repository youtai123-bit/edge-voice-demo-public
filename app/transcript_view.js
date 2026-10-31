/* Clearing this browser display never calls a backend deletion API. */
(function (root) {
  'use strict';
  class TranscriptBuffer {
    constructor(storage, key = 'edgevoice-display-v1') {
      this.storage = storage; this.key = key; this.rows = new Map(); this.cutoffs = {};
      try {
        const saved = JSON.parse(storage.getItem(key) || '{}'); this.cutoffs = saved.cutoffs || {};
        for (const row of (saved.rows || []).slice(-500)) {
          if (typeof row.text === 'string' && typeof row.key === 'string') this.rows.set(row.key, row);
        }
      } catch (_) { /* Backend archives are independent of browser storage. */ }
    }
    session(state) { return state.session_id || String(state.started_at || 'legacy'); }
    persist() {
      while (this.rows.size > 500) this.rows.delete(this.rows.keys().next().value);
      try { this.storage.setItem(this.key, JSON.stringify({rows: [...this.rows.values()], cutoffs: this.cutoffs})); }
      catch (_) { /* Keep the in-memory view if browser storage is unavailable. */ }
    }
    ingest(state) {
      const session = this.session(state), cutoff = this.cutoffs[session] || {index: -1, generation: -1};
      let changed = false;
      for (const item of (state.transcript || [])) {
        const itemSession = item.session_id || session;
        const itemCutoff = this.cutoffs[itemSession] || {index: -1, generation: -1};
        if (item.index <= itemCutoff.index || (item.preview_generation != null && item.preview_generation <= itemCutoff.generation)) continue;
        const row = {key: item.record_id || itemSession + ':' + item.index, session: itemSession,
          index: item.index, text: item.text,
          language: item.language_label || item.language || '', latency_ms: item.latency_ms,
          user_corrected: Boolean(item.user_corrected)};
        if (JSON.stringify(this.rows.get(row.key)) !== JSON.stringify(row)) { this.rows.set(row.key, row); changed = true; }
      }
      if (changed) this.persist();
      const draft = state.partial_active && state.partial_generation > cutoff.generation
        ? {text: state.partial_text || '正在听…', generation: state.partial_generation} : null;
      return {rows: [...this.rows.values()], draft};
    }
    clear(state) {
      for (const row of [...this.rows.values(), ...(state.transcript || []).map(item => ({
        session: item.session_id || this.session(state), index: item.index}))]) {
        const previous = this.cutoffs[row.session] || {index: -1, generation: -1};
        this.cutoffs[row.session] = {...previous, index: Math.max(previous.index, row.index)};
      }
      const session = this.session(state), previous = this.cutoffs[session] || {index: -1, generation: -1};
      this.cutoffs[session] = {index: previous.index,
        generation: Math.max(previous.generation, state.partial_generation || 0)};
      this.rows.clear(); this.persist();
    }
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = {TranscriptBuffer};
  else root.TranscriptBuffer = TranscriptBuffer;
})(typeof globalThis !== 'undefined' ? globalThis : this);
