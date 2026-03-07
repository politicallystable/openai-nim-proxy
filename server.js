// server.js - OpenAI to NVIDIA NIM API Proxy (Refined & Complete)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 CONFIGURATION TOGGLES
const FORCE_STREAMING = true; // Set to false to disable streaming
const DEFAULT_TEMPERATURE = 0.7;
const DEFAULT_MAX_TOKENS = 4096;
const DEFAULT_TIMEOUT = 180000; // 3 minutes (180 seconds)

// 🎯 COMPLETE MODEL MAPPING - All NVIDIA NIM Models (March 2026)
const MODEL_MAPPING = {
  // === GPT MODELS (Mapped to Best Alternatives) ===
  'gpt-3.5-turbo': 'meta/llama-3.1-8b-instruct',
  'gpt-4': 'meta/llama-3.3-70b-instruct',
  'gpt-4-turbo': 'meta/llama-3.1-405b-instruct',
  'gpt-4o': 'meta/llama-3.3-70b-instruct',
  'gpt-4o-mini': 'meta/llama-3.1-8b-instruct',
  
  // === DEEPSEEK MODELS (Reasoning & Coding) ===
  'deepseek-r1': 'deepseek-ai/deepseek-r1',
  'deepseek-v3.1': 'deepseek-ai/deepseek-v3_1',
  'deepseek-v3.2': 'deepseek-ai/deepseek-v3_2',
  'deepseek-r1-distill-qwen-32b': 'deepseek-ai/deepseek-r1-distill-qwen-32b',
  'deepseek-r1-distill-qwen-14b': 'deepseek-ai/deepseek-r1-distill-qwen-14b',
  'deepseek-r1-distill-qwen-7b': 'deepseek-ai/deepseek-r1-distill-qwen-7b',
  'deepseek-r1-distill-llama-70b': 'deepseek-ai/deepseek-r1-distill-llama-70b',
  'deepseek-r1-distill-llama-8b': 'deepseek-ai/deepseek-r1-distill-llama-8b',
  
  // === KIMI MODELS (256K Context!) ===
  'kimi': 'moonshotai/kimi-k2-instruct',
  'kimi-k2': 'moonshotai/kimi-k2-instruct',
  'kimi-k2-instruct': 'moonshotai/kimi-k2-instruct',
  'kimi-k2-instruct-0905': 'moonshotai/kimi-k2-instruct-0905',
  'kimi-k2-thinking': 'moonshotai/kimi-k2-thinking',
  'kimi-2.5': 'moonshotai/kimi-k2.5',
  'kimi-k2.5': 'moonshotai/kimi-k2.5',
  
  // === GLM MODELS (Zhipu AI - Coding & Reasoning!) ===
  'glm-5': 'z-ai/glm5', // ⭐⭐⭐ NEW! 744B params, beats GPT-5.2!
  'glm5': 'z-ai/glm5',
  'glm-4.7': 'z-ai/glm4.7',
  'glm4.7': 'z-ai/glm4.7',
  'glm-4': 'z-ai/glm4.7',
  
  // === QWEN REASONING MODELS ===
  'qwen-thinking': 'qwen/qwen3-next-80b-a3b-thinking',
  'qwen3-next-thinking': 'qwen/qwen3-next-80b-a3b-thinking',
  'qwen3-next-80b-thinking': 'qwen/qwen3-next-80b-a3b-thinking',
  'qwen3-next-instruct': 'qwen/qwen3-next-80b-a3b-instruct',
  'qwen3-235b': 'qwen/qwen3-235b-a22b',
  'qwen3-30b': 'qwen/qwen3-30b-a3b',
  'qwq-32b': 'qwen/qwq-32b-preview',
  
  // === META LLAMA MODELS ===
  'llama-3.1-405b': 'meta/llama-3.1-405b-instruct',
  'llama-3.1-70b': 'meta/llama-3.1-70b-instruct',
  'llama-3.1-8b': 'meta/llama-3.1-8b-instruct',
  'llama-3.2-1b': 'meta/llama-3.2-1b-instruct',
  'llama-3.2-3b': 'meta/llama-3.2-3b-instruct',
  'llama-3.3-70b': 'meta/llama-3.3-70b-instruct',
  
  // === NVIDIA NEMOTRON MODELS ===
  'nemotron-ultra': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'nemotron-70b': 'nvidia/llama-3.1-nemotron-70b-instruct',
  'nemotron-nano': 'nvidia/nemotron-3-nano-30b-a3b',
  'nemotron-super': 'nvidia/llama-3.3-nemotron-super-49b-v1',
  
  // === GOOGLE GEMMA MODELS ===
  'gemma-27b': 'google/gemma-2-27b-it',
  'gemma-9b': 'google/gemma-2-9b-it',
  'gemma-2b': 'google/gemma-2-2b-it',
  
  // === QWEN STANDARD MODELS ===
  'qwen-72b': 'qwen/qwen2.5-72b-instruct',
  'qwen-32b': 'qwen/qwen2.5-32b-instruct',
  'qwen-14b': 'qwen/qwen2.5-14b-instruct',
  'qwen-7b': 'qwen/qwen2.5-7b-instruct',
  
  // === MISTRAL/MIXTRAL MODELS ===
  'mixtral-8x7b': 'mistralai/mixtral-8x7b-instruct-v0.1',
  'mixtral-8x22b': 'mistralai/mixtral-8x22b-instruct-v0.1',
  'mistral-7b': 'mistralai/mistral-7b-instruct-v0.3',
  
  // === MICROSOFT PHI MODELS ===
  'phi-3': 'microsoft/phi-3-medium-4k-instruct',
  'phi-3-small': 'microsoft/phi-3-small-8k-instruct',
  'phi-3-mini': 'microsoft/phi-3-mini-128k-instruct',
  
  // === MINIMAX MODELS (Ultra-fast!) ===
  'minimax': 'minimaxai/minimax-m2.1',
  'minimax-m2.1': 'minimaxai/minimax-m2.1',
  'minimax-2.1': 'minimaxai/minimax-m2.1',
  
  // === CLAUDE MODELS (Mapped to Best Alternatives) ===
  'claude-3-opus': 'meta/llama-3.1-405b-instruct',
  'claude-3-sonnet': 'meta/llama-3.3-70b-instruct',
  'claude-3-haiku': 'meta/llama-3.1-8b-instruct'
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy',
    version: '2.0',
    streaming: FORCE_STREAMING,
    total_models: Object.keys(MODEL_MAPPING).length,
    featured_models: {
      newest: 'glm-5 (744B params, beats GPT-5.2!)',
      fastest: 'minimax (150 tokens/sec)',
      smartest: 'nemotron-ultra (253B params)',
      longest_context: 'kimi-2.5 (256K tokens)'
    }
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Model selection with fallback
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      nimModel = model; // Try using the model name directly
    }
    
    console.log(`[${new Date().toISOString()}] ${model} → ${nimModel}`);
    
    // Smart timeout based on model type
    const isKimiModel = nimModel.includes('kimi');
    const isGLMModel = nimModel.includes('glm');
    const timeoutDuration = (isKimiModel || isGLMModel) ? 0 : DEFAULT_TIMEOUT;
    
    // Build request
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || DEFAULT_TEMPERATURE,
      max_tokens: max_tokens || DEFAULT_MAX_TOKENS,
      stream: FORCE_STREAMING
    };
    
    // Make API request
    const response = await axios.post(`${NIM_API_BASE}/v1/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: FORCE_STREAMING ? 'stream' : 'json',
      timeout: timeoutDuration
    });
    
    // Handle streaming
    if (FORCE_STREAMING && response.data.on) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            res.write(line + '\n');
          }
        });
      });
      
      response.data.on('end', () => {
        console.log(`[${new Date().toISOString()}] ✓ ${model} completed`);
        res.end();
      });
      
      response.data.on('error', (err) => {
        console.error(`[${new Date().toISOString()}] ✗ ${model} error:`, err.message);
        res.end();
      });
    } else {
      // Non-streaming response
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => ({
          index: choice.index,
          message: {
            role: choice.message.role,
            content: choice.message.content || ''
          },
          finish_reason: choice.finish_reason
        })),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      console.log(`[${new Date().toISOString()}] ✓ ${model} completed`);
      res.json(openaiResponse);
    }
    
  } catch (error) {
    const status = error.response?.status || 500;
    const errorDetail = error.response?.data?.detail || error.message || 'Internal server error';
    
    console.error(`[${new Date().toISOString()}] ✗ Error ${status}:`, errorDetail);
    
    res.status(status).json({
      error: {
        message: `NVIDIA API Error: ${errorDetail}`,
        type: 'invalid_request_error',
        code: status,
        model_attempted: req.body?.model
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`
╔════════════════════════════════════════════════════════════════╗
║  OpenAI → NVIDIA NIM Proxy v2.0                                ║
╠════════════════════════════════════════════════════════════════╣
║  Port: ${PORT}                                                      ║
║  Health: http://localhost:${PORT}/health                        ║
║  Models: ${Object.keys(MODEL_MAPPING).length} available                                           ║
║  Streaming: ${FORCE_STREAMING ? 'ENABLED ✓' : 'DISABLED ✗'}                                    ║
╠════════════════════════════════════════════════════════════════╣
║  🆕 NEW: GLM-5 (744B params, beats GPT-5.2!)                   ║
║  ⚡ FAST: MiniMax (150 tokens/sec)                             ║
║  🧠 SMART: Nemotron Ultra (253B params)                        ║
║  📚 CONTEXT: Kimi 2.5 (256K tokens)                            ║
╚════════════════════════════════════════════════════════════════╝
  `);
});
