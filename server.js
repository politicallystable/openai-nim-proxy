// server.js - OpenAI to NVIDIA NIM API Proxy (ALL MODELS)
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

// COMPLETE MODEL MAPPING - ALL AVAILABLE MODELS
const MODEL_MAPPING = {
  // GPT Models (mapped to best alternatives)
  'gpt-3.5-turbo': 'meta/llama-3.1-8b-instruct',
  'gpt-4': 'meta/llama-3.3-70b-instruct',
  'gpt-4-turbo': 'meta/llama-3.1-405b-instruct',
  'gpt-4o': 'meta/llama-3.3-70b-instruct',
  'gpt-4o-mini': 'meta/llama-3.1-8b-instruct',
  
  // DeepSeek Models (Reasoning & Coding)
  'deepseek-r1': 'deepseek-ai/deepseek-r1',
  'deepseek-v3.1': 'deepseek-ai/deepseek-v3_1',
  'deepseek-v3.2': 'deepseek-ai/deepseek-v3_2',
  'deepseek-r1-distill-qwen-32b': 'deepseek-ai/deepseek-r1-distill-qwen-32b',
  'deepseek-r1-distill-qwen-14b': 'deepseek-ai/deepseek-r1-distill-qwen-14b',
  'deepseek-r1-distill-qwen-7b': 'deepseek-ai/deepseek-r1-distill-qwen-7b',
  'deepseek-r1-distill-llama-70b': 'deepseek-ai/deepseek-r1-distill-llama-70b',
  'deepseek-r1-distill-llama-8b': 'deepseek-ai/deepseek-r1-distill-llama-8b',
  
  // Kimi Models (256K context! - UPDATED with Kimi 2.5!)
  'kimi': 'moonshotai/kimi-k2-instruct',
  'kimi-k2': 'moonshotai/kimi-k2-instruct',
  'kimi-k2-instruct': 'moonshotai/kimi-k2-instruct',
  'kimi-k2-instruct-0905': 'moonshotai/kimi-k2-instruct-0905',
  'kimi-k2-thinking': 'moonshotai/kimi-k2-thinking',
  'kimi-k2.5': 'moonshotai/kimi-k2.5',
  'kimi-2.5': 'moonshotai/kimi-k2.5',
  
  // Qwen Reasoning Models (Powerful! - Some may be degraded)
  'qwen-thinking': 'qwen/qwen3-next-80b-a3b-thinking', // ⚠️ May be degraded
  'qwen3-next-thinking': 'qwen/qwen3-next-80b-a3b-thinking', // ⚠️ May be degraded
  'qwen3-next-80b-thinking': 'qwen/qwen3-next-80b-a3b-thinking', // ⚠️ May be degraded
  'qwen3-next-instruct': 'qwen/qwen3-next-80b-a3b-instruct', // ✅ Should work
  'qwen3-235b': 'qwen/qwen3-235b-a22b', // ⚠️ May be degraded
  'qwen3-30b': 'qwen/qwen3-30b-a3b', // ⚠️ May be degraded
  'qwq-32b': 'qwen/qwq-32b-preview', // ⚠️ May be degraded
  
  // GLM Models (Zhipu AI - Coding & Reasoning!)
  'glm-4.7': 'z-ai/glm4_7', // ⭐⭐⭐ NEW! 400B params, 200K context, amazing for coding!
  'glm4.7': 'z-ai/glm4_7',
  
  // Meta Llama Models
  'llama-3.1-405b': 'meta/llama-3.1-405b-instruct',
  'llama-3.1-70b': 'meta/llama-3.1-70b-instruct',
  'llama-3.1-8b': 'meta/llama-3.1-8b-instruct',
  'llama-3.2-1b': 'meta/llama-3.2-1b-instruct',
  'llama-3.2-3b': 'meta/llama-3.2-3b-instruct',
  'llama-3.3-70b': 'meta/llama-3.3-70b-instruct',
  
  // NVIDIA Nemotron Models
  'nemotron-ultra': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'nemotron-70b': 'nvidia/llama-3.1-nemotron-70b-instruct',
  'nemotron-nano': 'nvidia/nemotron-3-nano-30b-a3b',
  'nemotron-super': 'nvidia/llama-3.3-nemotron-super-49b-v1',
  
  // Google Gemma Models
  'gemma-27b': 'google/gemma-2-27b-it',
  'gemma-9b': 'google/gemma-2-9b-it',
  'gemma-2b': 'google/gemma-2-2b-it',
  
  // Qwen Standard Models
  'qwen-72b': 'qwen/qwen2.5-72b-instruct',
  'qwen-32b': 'qwen/qwen2.5-32b-instruct',
  'qwen-14b': 'qwen/qwen2.5-14b-instruct',
  'qwen-7b': 'qwen/qwen2.5-7b-instruct',
  
  // Mistral/Mixtral Models
  'mixtral-8x7b': 'mistralai/mixtral-8x7b-instruct-v0.1',
  'mixtral-8x22b': 'mistralai/mixtral-8x22b-instruct-v0.1',
  'mistral-7b': 'mistralai/mistral-7b-instruct-v0.3',
  
  // Microsoft Phi Models
  'phi-3': 'microsoft/phi-3-medium-4k-instruct',
  'phi-3-small': 'microsoft/phi-3-small-8k-instruct',
  'phi-3-mini': 'microsoft/phi-3-mini-128k-instruct',
  
  // Claude Models (mapped to best alternatives)
  'claude-3-opus': 'meta/llama-3.1-405b-instruct',
  'claude-3-sonnet': 'meta/llama-3.3-70b-instruct',
  'claude-3-haiku': 'meta/llama-3.1-8b-instruct'
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy (Complete)', 
    total_models: Object.keys(MODEL_MAPPING).length,
    note: 'All models included - get proper API key for premium models'
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
    
    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      // If model not in mapping, try using it directly
      nimModel = model;
    }
    
    console.log(`[REQUEST] User requested: ${model} → Using: ${nimModel}`);
    
    // Set timeout based on model type (unlimited for Kimi models)
    const isKimiModel = nimModel.includes('kimi');
    const timeoutDuration = isKimiModel ? 0 : 180000; // 0 = unlimited, 180000 = 3 minutes
    
    // Transform OpenAI request to NIM format
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.7,
      max_tokens: max_tokens || 4096,
      stream: stream !== false // Force streaming for better reliability
    };
    
    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/v1/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: timeoutDuration // Unlimited for Kimi, 3 minutes for others
    });
    
    if (stream) {
      // Handle streaming response
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
        console.log(`[SUCCESS] Streaming completed for ${model}`);
        res.end();
      });
      
      response.data.on('error', (err) => {
        console.error('[ERROR] Stream error:', err);
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
      
      console.log(`[SUCCESS] Completed request for ${model}`);
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('[ERROR] Proxy error:', error.response?.status, error.message);
    console.error('[ERROR] Model attempted:', req.body?.model);
    
    const status = error.response?.status || 500;
    const errorMessage = error.response?.data?.detail || error.message || 'Internal server error';
    
    res.status(status).json({
      error: {
        message: `NVIDIA API Error: ${errorMessage}`,
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

app.listen(PORT, () => {
  console.log(`========================================`);
  console.log(`OpenAI to NVIDIA NIM Proxy (Complete)`);
  console.log(`Running on port ${PORT}`);
  console.log(`Health: http://localhost:${PORT}/health`);
  console.log(`Models: ${Object.keys(MODEL_MAPPING).length} available`);
  console.log(`========================================`);
});
