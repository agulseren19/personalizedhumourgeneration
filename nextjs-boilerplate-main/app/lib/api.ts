import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 seconds for humor generation (increased for CrewAI black card generation which takes ~28-30 seconds)
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor to include auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('cah_token');
    console.log('🔍 API Request Interceptor:', {
      url: config.url,
      method: config.method,
      hasToken: !!token,
      tokenPreview: token ? `${token.substring(0, 20)}...` : 'None',
      tokenLength: token ? token.length : 0,
      tokenValue: token,
      tokenTrimmed: token ? token.trim() : 'N/A',
      headers: config.headers
    });
    
    if (token && token.trim() !== '') {
      config.headers.Authorization = `Bearer ${token}`;
      console.log('🔍 Added Authorization header:', `Bearer ${token.substring(0, 20)}...`);
      console.log('🔍 Full Authorization header:', config.headers.Authorization);
      console.log('🔍 Final headers:', config.headers);
    } else {
      console.log('🔍 No valid token found, request will be unauthenticated');
      console.log('🔍 Token details:', {
        exists: !!token,
        length: token ? token.length : 0,
        value: token,
        trimmed: token ? token.trim() : 'N/A',
        isEmpty: token ? token.trim() === '' : true
      });
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Types for API responses
export interface Persona {
  id: string;
  name: string;
  description: string;
  expertise: string[];
  humor_style: string;
}

export interface Generation {
  id: string;
  text: string;
  persona_name: string;
  model_used: string;
  score: number;
  humor_score: number;
  creativity_score: number;
  appropriateness_score: number;
  context_relevance_score?: number;
  surprise_index?: number;  // Add surprise index property
  is_safe: boolean;
  toxicity_score: number;
  reasoning: string;
}

export interface HumorResponse {
  success: boolean;
  generations: Generation[];
  recommended_personas: string[];
  generation_time: number;
  error?: string;
}

export interface UserAnalytics {
  total_interactions: number;
  average_score: number;
  liked_personas: string[];
  disliked_personas: string[];
  persona_performance: Record<string, {
    avg_score: number;
    interaction_count: number;
    status: 'liked' | 'disliked' | 'neutral';
  }>;
}

// Multiplayer Game Types
export interface Player {
  user_id: string;  // Keep as string for now to avoid breaking changes
  username: string;
  score: number;
  is_host: boolean;
  is_judge: boolean;
  connected: boolean;
  has_submitted: boolean;
}

export interface GameRound {
  round_number: number;
  black_card: string;
  phase: 'card_submission' | 'judging' | 'results';
  judge_id: string;  // Keep as string for now to avoid breaking changes
  judge_username: string;
  submissions_count: number;
  my_submission?: string;
  submissions?: Array<{ card: string; player_username: string }>;
  judge_can_see_submissions?: boolean;
  submissions_visible?: boolean;
}

export interface GameState {
  game_id: string;
  status: 'waiting' | 'starting' | 'in_progress' | 'judging' | 'finished';
  players: Player[];
  my_hand?: string[];
  is_my_turn?: boolean;
  current_round?: GameRound;
  round_history?: Array<{
    round_number: number;
    black_card: string;
    winner_username: string;
    winning_card: string;
  }>;
  max_players: number;
  max_score: number;
  round_timer?: number;
}

export interface CreateGameRequest {
  max_score: number;
  max_players: number;
  round_timer: number;
}

export interface GenerateHumorRequest {
  context: string;
  audience: string;
  topic: string;
  user_id: string;  // Keep as string for now to avoid breaking changes
  card_type: 'white' | 'black' | 'game' | 'multiplayer';
}

export interface FeedbackRequest {
  user_id: string;  // Keep as string for now to avoid breaking changes
  persona_name: string;
  feedback_score: number;
  context: string;
  response_text: string;
  topic: string;
  audience: string;
}

// API functions
export const cahApi = {
  // Health check
  async healthCheck() {
    const response = await api.get('/');
    return response.data;
  },

  // Get all personas
  async getPersonas(): Promise<{ success: boolean; personas: Persona[] }> {
    const response = await api.get('/personas');
    return response.data;
  },

  // Generate humor
  async generateHumor(request: GenerateHumorRequest): Promise<HumorResponse> {
    const response = await api.post('/generate', request);
    return response.data;
  },

  // Submit feedback
  async submitFeedback(feedback: FeedbackRequest): Promise<{ success: boolean; message: string }> {
    const response = await api.post('/feedback', feedback);
    return response.data;
  },

  // Get user analytics
  async getUserAnalytics(userId: string): Promise<{ success: boolean; analytics: UserAnalytics }> {
    const response = await api.get(`/analytics/${userId}`);
    return response.data;
  },

  // Get persona recommendations
  async getPersonaRecommendations(
    userId: string,
    context: string,
    audience: string = 'general'
  ): Promise<{ success: boolean; recommendations: string[] }> {
    const response = await api.get(`/recommendations/${userId}`, {
      params: { context, audience }
    });
    return response.data;
  },

  // Check content safety
  async checkContentSafety(text: string): Promise<{
    is_safe: boolean;
    toxicity_score: number;
    sanitized_content: string;
  }> {
    const response = await api.post('/content-filter', { text });
    return response.data;
  },

  // MULTIPLAYER GAME FUNCTIONS
  // Create a new game
  async createGame(request: CreateGameRequest): Promise<{
    success: boolean;
    game_id: string;
    message: string;
    game_state: any;
    error?: string;
  }> {
    const token = typeof window !== 'undefined' ? localStorage.getItem('cah_token') : null;
    const response = await api.post('/multiplayer/create-game', request, {
      headers: token ? { Authorization: `Bearer ${token}` } : {}
    });
    return response.data;
  },

  // Join an existing game
  async joinGame(gameId: string, userId: string): Promise<{
    success: boolean;
    message: string;
    game_state: any;
    error?: string;
  }> {
    const token = typeof window !== 'undefined' ? localStorage.getItem('cah_token') : null;
    const response = await api.post(`/multiplayer/join-game/${gameId}`, undefined, {
      headers: token ? { Authorization: `Bearer ${token}` } : {}
    });
    return response.data;
  },

  // Start a game (host only)
  async startGame(gameId: string, hostUserId: string): Promise<{
    success: boolean;
    message: string;
    status: string;
  }> {
    const token = typeof window !== 'undefined' ? localStorage.getItem('cah_token') : null;
    const response = await api.post(`/multiplayer/start-game/${gameId}`, undefined, {
      headers: token ? { Authorization: `Bearer ${token}` } : {}
    });
    return response.data;
  },

  // Submit a white card
  async submitCard(gameId: string, userId: string, whiteCard: string): Promise<{
    success: boolean;
    message: string;
  }> {
    const token = typeof window !== 'undefined' ? localStorage.getItem('cah_token') : null;
    const response = await api.post(`/multiplayer/submit-card/${gameId}?white_card=${encodeURIComponent(whiteCard)}`, undefined, {
      headers: token ? { Authorization: `Bearer ${token}` } : {}
    });
    return response.data;
  },

  // Judge the round (judge only)
  async judgeRound(gameId: string, judgeUserId: string, winningCard: string): Promise<{
    success: boolean;
    message: string;
    winner_id: string;
  }> {
    const token = typeof window !== 'undefined' ? localStorage.getItem('cah_token') : null;
    const response = await api.post(`/multiplayer/judge-round/${gameId}?winning_user_id=${judgeUserId}&winning_card=${encodeURIComponent(winningCard)}`, undefined, {
      headers: token ? { Authorization: `Bearer ${token}` } : {}
    });
    return response.data;
  },

  // Get current game state
  async getGameState(gameId: string, userId: string): Promise<{
    success: boolean;
    game_state: GameState;
  }> {
    const token = typeof window !== 'undefined' ? localStorage.getItem('cah_token') : null;
    const response = await api.get(`/multiplayer/game-state/${gameId}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {}
    });
    return response.data;
  },

  // Get all games (for debugging)
  async getAllGames(): Promise<{
    success: boolean;
    games: Array<{
      game_id: string;
      status: string;
      player_count: number;
      created_at: string;
    }>;
  }> {
    const token = typeof window !== 'undefined' ? localStorage.getItem('cah_token') : null;
    const response = await api.get('/multiplayer/games', {
      headers: token ? { Authorization: `Bearer ${token}` } : {}
    });
    return response.data;
  },

  // Leave a multiplayer game
  async leaveGame(gameId: string): Promise<{
    success: boolean;
    message: string;
  }> {
    const token = typeof window !== 'undefined' ? localStorage.getItem('cah_token') : null;
    const response = await api.post(`/multiplayer/leave-game/${gameId}`, undefined, {
      headers: token ? { Authorization: `Bearer ${token}` } : {}
    });
    return response.data;
  },

  // Get player's current hand
  async getPlayerHand(gameId: string): Promise<{
    success: boolean;
    hand: string[];
    hand_size: number;
  }> {
    const token = typeof window !== 'undefined' ? localStorage.getItem('cah_token') : null;
    const response = await api.get(`/multiplayer/player-hand/${gameId}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {}
    });
    return response.data;
  },

  // Replenish player's hand
  async replenishHand(gameId: string): Promise<{
    success: boolean;
    message: string;
    hand: string[];
    hand_size: number;
  }> {
    const token = typeof window !== 'undefined' ? localStorage.getItem('cah_token') : null;
    const response = await api.post(`/multiplayer/replenish-hand/${gameId}`, undefined, {
      headers: token ? { Authorization: `Bearer ${token}` } : {}
    });
    return response.data;
  },

  // Debug game cards and player hands
  async debugGameCards(gameId: string): Promise<{
    success: boolean;
    debug_info: {
      game_id: string;
      status: string;
      players: Array<{
        user_id: string;
        username: string;
        email: string;
        is_host: boolean;
        is_judge: boolean;
        score: number;
        hand_size: number;
        hand_sample: string[];
        has_hand: boolean;
        hand_type: string;
      }>;
      current_round: any;
      debug_info: {
        total_players: number;
        cards_prepared: boolean;
        black_cards_count: number;
      };
    };
  }> {
    const token = typeof window !== 'undefined' ? localStorage.getItem('cah_token') : null;
    const response = await api.get(`/multiplayer/debug/game-cards/${gameId}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {}
    });
    return response.data;
  }

};

// Utility function to generate user ID
export const generateUserId = (): string => {
  const stored = localStorage.getItem('cah_user_id');
  if (stored) return stored;
  
  const newId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  localStorage.setItem('cah_user_id', newId);
  return newId;
};

// Utility function to generate consistent user ID from username (matches backend)
export const generateUserIdFromUsername = (username: string): string => {
  // Simple hash function to match backend logic
  let hash = 0;
  for (let i = 0; i < username.length; i++) {
    const char = username.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return `user_${Math.abs(hash) % 10000}`;
};

// Error handler
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    throw error;
  }
); 