// Roadmap site — audience toggle + Mermaid init + smooth scroll
(function () {
  'use strict';

  const STORAGE_KEY = 'roadmap-audience';
  const DEFAULT_LEVEL = 'researcher';

  function getLevel() {
    try {
      return localStorage.getItem(STORAGE_KEY) || DEFAULT_LEVEL;
    } catch (e) {
      return DEFAULT_LEVEL;
    }
  }

  function setLevel(level) {
    try { localStorage.setItem(STORAGE_KEY, level); } catch (e) { /* ignore */ }
    applyLevel(level);
  }

  function applyLevel(level) {
    document.documentElement.setAttribute('data-audience-active', level);
    document.querySelectorAll('.audience-btn').forEach(btn => {
      btn.classList.toggle('is-active', btn.dataset.level === level);
      btn.setAttribute('aria-pressed', String(btn.dataset.level === level));
    });
  }

  function initAudienceToggle() {
    const buttons = document.querySelectorAll('.audience-btn');
    if (!buttons.length) return;
    buttons.forEach(btn => {
      btn.addEventListener('click', () => setLevel(btn.dataset.level));
    });
    applyLevel(getLevel());
  }

  function initMermaid() {
    if (typeof window.mermaid === 'undefined') return;
    const root = document.documentElement;
    const styles = getComputedStyle(root);
    const bg = styles.getPropertyValue('--bg-2').trim() || '#0E260B';
    const accent = styles.getPropertyValue('--accent-primary').trim() || '#7BA762';
    const accentLight = styles.getPropertyValue('--accent-tertiary').trim() || '#9DC288';
    const text = styles.getPropertyValue('--text-primary').trim() || '#f0f5ec';
    const border = '#7BA762';

    window.mermaid.initialize({
      startOnLoad: true,
      theme: 'base',
      themeVariables: {
        background: bg,
        primaryColor: bg,
        primaryTextColor: text,
        primaryBorderColor: accent,
        lineColor: accent,
        secondaryColor: '#1a3a15',
        tertiaryColor: '#0E260B',
        nodeBorder: border,
        clusterBkg: '#1a3a15',
        edgeLabelBackground: bg,
        fontFamily: 'JetBrains Mono, monospace',
      },
      flowchart: { curve: 'basis', padding: 18 },
      sequence: { actorMargin: 50 },
    });
  }

  function initSmoothAnchors() {
    document.querySelectorAll('a[href^="#"]').forEach(a => {
      a.addEventListener('click', e => {
        const id = a.getAttribute('href').slice(1);
        if (!id) return;
        const el = document.getElementById(id);
        if (!el) return;
        e.preventDefault();
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        history.replaceState(null, '', '#' + id);
      });
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    initAudienceToggle();
    initMermaid();
    initSmoothAnchors();
  });
})();
