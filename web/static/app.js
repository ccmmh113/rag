const { createApp } = Vue;

createApp({
  data() {
    return {
      view: 'chat',
      loading: false,
      question: '',
      messages: [],
      citations: [],
      lastTrace: {},
      status: {},
      traces: [],
      traceStats: {},
      memories: [],
      memoryForm: { content: '', entry_type: 'note' },
      configForm: {
        dense_top_k: 50,
        sparse_top_k: 50,
        final_top_k: 50,
        reranker_top_k: 8,
        context_max_tokens: 3000,
        dense_weight: 0.9,
        sparse_weight: 0.1,
        fusion: 'rrf',
        index_type: 'flat_ip',
        use_reranker: true,
      },
    };
  },
  computed: {
    pageTitle() {
      return {
        chat: 'Chat',
        traces: 'Trace',
        memory: 'Memory',
        config: 'Config',
      }[this.view];
    },
    pageSubtitle() {
      return {
        chat: 'Ask questions and inspect citations in real time.',
        traces: 'Review retrieval, rerank, latency and token behavior.',
        memory: 'Manage notes and preferences used by the RAG runtime.',
        config: 'Tune retrieval and context parameters before the next query.',
      }[this.view];
    },
  },
  async mounted() {
    await this.refreshStatus();
    await this.loadConfig();
  },
  methods: {
    async api(path, options = {}) {
      const response = await fetch(path, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || `Request failed: ${response.status}`);
      }
      return data;
    },
    async refreshStatus() {
      this.status = await this.api('/api/status');
    },
    async loadConfig() {
      const cfg = await this.api('/api/config');
      this.configForm = {
        dense_top_k: cfg.retrieval.dense_top_k,
        sparse_top_k: cfg.retrieval.sparse_top_k,
        final_top_k: cfg.retrieval.final_top_k,
        reranker_top_k: cfg.reranker.top_k,
        context_max_tokens: cfg.context.max_tokens,
        dense_weight: cfg.retrieval.dense_weight,
        sparse_weight: cfg.retrieval.sparse_weight,
        fusion: cfg.retrieval.fusion,
        index_type: cfg.index.index_type,
        use_reranker: cfg.use_reranker,
      };
    },
    async sendQuestion() {
      const text = this.question.trim();
      if (!text || this.loading) return;
      this.loading = true;
      this.question = '';
      this.messages.push({ id: crypto.randomUUID(), role: 'user', content: text });
      try {
        const data = await this.api('/api/chat', {
          method: 'POST',
          body: JSON.stringify({ question: text }),
        });
        this.messages.push({ id: crypto.randomUUID(), role: 'assistant', content: data.answer });
        this.citations = data.citations || [];
        this.lastTrace = data.trace || {};
        await this.refreshStatus();
      } catch (error) {
        this.messages.push({ id: crypto.randomUUID(), role: 'assistant', content: error.message });
      } finally {
        this.loading = false;
      }
    },
    async openTraces() {
      this.view = 'traces';
      const data = await this.api('/api/traces?limit=20');
      this.traces = data.items || [];
      this.traceStats = data;
    },
    async openMemory() {
      this.view = 'memory';
      const data = await this.api('/api/memories');
      this.memories = data.items || [];
    },
    async saveMemory() {
      const content = this.memoryForm.content.trim();
      if (!content) return;
      await this.api('/api/memories', {
        method: 'POST',
        body: JSON.stringify({ content, entry_type: this.memoryForm.entry_type }),
      });
      this.memoryForm.content = '';
      await this.openMemory();
    },
    async deleteMemory(id) {
      await this.api(`/api/memories/${id}`, { method: 'DELETE' });
      await this.openMemory();
    },
    async saveConfig() {
      await this.api('/api/config', {
        method: 'PATCH',
        body: JSON.stringify(this.configForm),
      });
      await this.refreshStatus();
    },
    async buildIndex() {
      this.loading = true;
      try {
        this.status = await this.api('/api/index/build', { method: 'POST' });
      } finally {
        this.loading = false;
      }
    },
    formatScore(value) {
      if (value === null || value === undefined) return '-';
      return Number(value).toFixed(3);
    },
    formatMs(value) {
      if (value === null || value === undefined) return '-';
      return `${Number(value).toFixed(1)} ms`;
    },
    percent(value) {
      if (value === null || value === undefined) return '-';
      return `${(Number(value) * 100).toFixed(1)}%`;
    },
  },
}).mount('#app');
