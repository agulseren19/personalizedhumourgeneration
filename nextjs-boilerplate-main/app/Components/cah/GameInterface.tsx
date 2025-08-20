'use client';

import React, { useState, useEffect } from 'react';
import { cahApi, Generation, GenerateHumorRequest, GameState, Player, generateUserIdFromUsername } from '../../lib/api';
import toast from 'react-hot-toast';
import { Play, Sparkles, ThumbsUp, ThumbsDown, RotateCcw } from 'lucide-react';
import { useUser } from '../../contexts/UserContext';

interface GameInterfaceProps {
  userId: string;
}

export default function GameInterface({ userId }: GameInterfaceProps) {
  const { user, debugTokenStatus, clearTokenAndReload, setTokenManually, validateCurrentToken } = useUser();
  const [context, setContext] = useState('');
  const [audience, setAudience] = useState('friends');
  const [topic, setTopic] = useState('general');
  const [cardType, setCardType] = useState<'white' | 'black' | 'game' | 'multiplayer'>('multiplayer');
  const [generations, setGenerations] = useState<Generation[]>([]);
  const [loading, setLoading] = useState(false);
  const [generationTime, setGenerationTime] = useState(0);
  const [recommendations, setRecommendations] = useState<string[]>([]);
  const [round, setRound] = useState(1);
  const [selectedCard, setSelectedCard] = useState<string | null>(null);
  const [blackCardTopic, setBlackCardTopic] = useState('');
  const [blackCard, setBlackCard] = useState<any>(null);
  const [blackCardWhiteCards, setBlackCardWhiteCards] = useState<any[]>([]);
  const [gameCardRatings, setGameCardRatings] = useState<Record<string, number>>({});
  const [whiteCardRatings, setWhiteCardRatings] = useState<Record<string, number>>({});
  
  // Multiplayer game state
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [gameId, setGameId] = useState('');
  const [isInGame, setIsInGame] = useState(false);
  const [webSocket, setWebSocket] = useState<WebSocket | null>(null);
  const [gameUserId, setGameUserId] = useState<string>(''); // Store the user ID returned from backend
  const [refreshInterval, setRefreshInterval] = useState<NodeJS.Timeout | null>(null);
  const [isClient, setIsClient] = useState(false); // Track if component has mounted on client

  // Clear game state when switching card types
  useEffect(() => {
    if (cardType !== 'game') {
      setBlackCard(null);
      setBlackCardWhiteCards([]);
      setBlackCardTopic('');
      setGameCardRatings({});
    } else {
      // Clear regular generations when switching to game mode
      setGenerations([]);
      setContext('');
      setWhiteCardRatings({});
    }
    
    // Clear multiplayer state when leaving multiplayer mode
    if (cardType !== 'multiplayer') {
      setGameState(null);
      setIsInGame(false);
      setGameUserId(''); // Reset the game user ID
      if (webSocket) {
        webSocket.close();
        setWebSocket(null);
      }
      // Clear the refresh interval
      if (refreshInterval) {
        clearInterval(refreshInterval);
        setRefreshInterval(null);
      }
    }
  }, [cardType, webSocket, refreshInterval]);

  // Set isClient to true after component mounts
  useEffect(() => {
    setIsClient(true);
  }, []);

  // Cleanup effect for intervals and WebSocket
  useEffect(() => {
    return () => {
      if (refreshInterval) {
        clearInterval(refreshInterval);
      }
      if (webSocket) {
        webSocket.close();
      }
    };
  }, [refreshInterval, webSocket]);

  // Auto-refresh when phase changes to judging for judges
  useEffect(() => {
    if (gameState?.current_round?.phase === 'judging' && 
        gameState?.current_round?.judge_id === (gameUserId || userId) &&
        (!gameState.current_round.submissions || gameState.current_round.submissions.length === 0)) {
      // console.log('🎯 Auto-refreshing game state for judge in judging phase');
      // Small delay to allow backend to process
      const timer = setTimeout(() => {
        refreshGameState();
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [gameState?.current_round?.phase, gameState?.current_round?.judge_id, gameState?.current_round?.submissions]);

  // Polling mechanism for judges to ensure they see submissions
  useEffect(() => {
    if (gameState?.current_round?.judge_id === (gameUserId || userId) && 
        gameState?.current_round?.phase === 'judging') {
      // console.log('🎯 Setting up judge polling for submissions');
      const judgePollingInterval = setInterval(() => {
        if (gameState?.current_round && (!gameState.current_round.submissions || gameState.current_round.submissions.length === 0)) {
          // console.log('🎯 Judge polling: No submissions found, refreshing...');
          refreshGameState();
        }
      }, 3000); // Poll every 3 seconds
      
      return () => clearInterval(judgePollingInterval);
    }
  }, [gameState?.current_round?.judge_id, gameState?.current_round?.phase, gameState?.current_round?.submissions]);

  const audienceOptions = [
    { value: 'friends', label: '👥 Friends', description: 'Casual and fun' },
    { value: 'colleagues', label: '💼 Colleagues', description: 'Professional humor' },
    { value: 'family', label: '👨‍👩‍👧‍👦 Family', description: 'Family-friendly' },
    { value: 'general', label: '🌍 General', description: 'Broad audience' }
  ];

  const topicOptions = [
    { value: 'general', label: '🎯 General' },
    { value: 'work', label: '💼 Work' },
    { value: 'technology', label: '💻 Technology' },
    { value: 'relationships', label: '💕 Relationships' },
    { value: 'food', label: '🍕 Food' },
    { value: 'gaming', label: '🎮 Gaming' },
    { value: 'lifestyle', label: '🏠 Lifestyle' }
  ];

  // Sample contexts for inspiration - proper fill-in-the-blank black cards
  const sampleContexts = [
    "TSA guidelines now prohibit _____ on airplanes.",
    "What's the best excuse for being late to work? _____",
    "What did I find in my browser history? _____",
    "What's my secret guilty pleasure? _____",
    "What's the worst part about adult life? _____",
    "What's my most embarrassing moment? _____",
    "The CIA now interrogates suspects with _____.",
    "Next season on Man vs. Wild, Bear Grylls must survive in the depths of the Amazon with only _____.",
    "When Pharaoh remained unmoved, Moses called down a plague of _____.",
    "What's that sound? It's the sound of _____.",
    "In the new Disney Channel Original Movie, Hannah Montana struggles with _____ for the first time.",
    "What's my anti-drug? _____",
    "What's there a ton of in heaven? _____",
    "What would grandma find disturbing, yet oddly charming? _____",
    "What's the next Happy Meal toy? _____"
  ];

  const handleGenerate = async () => {
    if (!context.trim()) {
      toast.error('Please enter a context!');
      return;
    }

          // Use authenticated user ID if available, otherwise fall back to frontend ID
      const consistentUserId = user?.id || userId;
      

      
      if (!consistentUserId) {
        toast.error('User ID not available!');
        return;
      }

    setLoading(true);
    setGenerations([]);

    try {
      // Check if the context is already a black card (contains fill-in-the-blank)
      const isAlreadyBlackCard = context.trim().includes('_____');
      
      // Logic for determining effective card type:
      // 1. If user explicitly selected 'black' card type, generate black cards
      // 2. If context contains _____ and no explicit card type is selected, generate white cards
      // 3. Otherwise, use the selected cardType
      let effectiveCardType = cardType;
      
      if (cardType === 'black') {
        // User explicitly wants black cards - generate black cards regardless of context
        effectiveCardType = 'black';
      } else if (isAlreadyBlackCard && (cardType === 'white' || cardType === 'game' || cardType === 'multiplayer')) {
        // Context is already a black card and user wants responses to it
        effectiveCardType = 'white';
      }
      
      const request: GenerateHumorRequest = {
        context: context.trim(),
        audience,
        topic,
        user_id: consistentUserId,
        card_type: effectiveCardType
      };
      


      const result = await cahApi.generateHumor(request);
      
      if (result.success) {
        setGenerations(result.results || []);  // Backend'den gelen 'results' field'ı
        setRecommendations(result.recommended_personas || []);
        setGenerationTime(result.generation_time || 0);
        setRound(r => r + 1); // increment round
        
        // Show appropriate message based on what was generated
        if (isAlreadyBlackCard) {
          toast.success(`Generated ${result.results?.length || 0} white cards for your black card in ${result.generation_time?.toFixed(2) || '0.00'}s`);
        } else {
          toast.success(`Generated ${result.results?.length || 0} responses in ${result.generation_time?.toFixed(2) || '0.00'}s`);
        }
      } else {
        toast.error(result.error || 'Generation failed');
      }
    } catch (error) {
      console.error('Generation error:', error);
      toast.error('Failed to generate humor');
    } finally {
      setLoading(false);
    }
  };

  const handleFeedback = async (generation: Generation, score: number) => {
    try {
      // Use authenticated user ID if available, otherwise fall back to frontend ID
      const consistentUserId = user?.id || userId;
      
      await cahApi.submitFeedback({
        user_id: consistentUserId,
        persona_name: generation.persona_name,
        feedback_score: score,
        context: context,
        response_text: generation.text,
        topic: topic,
        audience: audience
      });

      // Update local state to show feedback was given
      setGenerations(prev => 
        prev.map(g => 
          g.id === generation.id 
            ? { ...g, userFeedback: score }
            : g
        )
      );

      const emoji = score >= 7 ? '👍' : score <= 4 ? '👎' : '🤔';
      toast.success(`${emoji} Feedback recorded! (${score}/10)`);
    } catch (error) {
      console.error('Feedback error:', error);
      toast.error('Failed to record feedback');
    }
  };

  const handleCardSelect = async (generation: Generation) => {
    setSelectedCard(generation.id);
    try {
      // Use authenticated user ID if available, otherwise fall back to frontend ID
      const consistentUserId = user?.id || userId;
      
      await cahApi.submitFeedback({
        user_id: consistentUserId,
        persona_name: generation.persona_name,
        feedback_score: 10,
        context: context,
        response_text: generation.text,
        topic: topic,
        audience: audience
      });
      toast.success(`Selected: "${generation.text}"`);
    } catch (error) {
      console.error('Feedback error:', error);
      toast.error('Failed to record selection');
    }
  };

  const handleGameCardRating = async (cardId: string, rating: number) => {
    setGameCardRatings(prev => ({ ...prev, [cardId]: rating }));
    
    // Find the card to get its details
    const card = blackCardWhiteCards.find(c => c.id === cardId);
    if (!card) return;

    try {
      // Use authenticated user ID if available, otherwise fall back to frontend ID
      const consistentUserId = user?.id || userId;
      
      await cahApi.submitFeedback({
        user_id: consistentUserId,
        persona_name: card.persona,
        feedback_score: rating,
        context: blackCard.text,
        response_text: card.text,
        topic: 'general',
        audience: 'general'
      });
      toast.success(`Rating saved: ${rating}/10`);
    } catch (error) {
      console.error('Game card feedback error:', error);
      toast.error('Failed to save rating');
    }
  };

  const handleWhiteCardRating = async (generation: Generation, rating: number) => {
    setWhiteCardRatings(prev => ({ ...prev, [generation.id]: rating }));
    
    try {
      // Use authenticated user ID if available, otherwise fall back to frontend ID
      const consistentUserId = user?.id || userId;
      
      await cahApi.submitFeedback({
        user_id: consistentUserId,
        persona_name: generation.persona_name,
        feedback_score: rating,
        context: context,
        response_text: generation.text,
        topic: topic,
        audience: audience
      });
      toast.success(`Rating saved: ${rating}/10`);
    } catch (error) {
      console.error('White card feedback error:', error);
      toast.error('Failed to save rating');
    }
  };

  const handleBlackCardGenerate = async () => {
    if (!blackCardTopic.trim()) {
      toast.error('Please enter a topic for black card!');
      return;
    }
    setLoading(true);
    try {
      // Use authenticated user ID if available, otherwise fall back to frontend ID
      const consistentUserId = user?.id || userId;
      
      // Generate black card
      const blackRequest: GenerateHumorRequest = {
        context: blackCardTopic.trim(),
        audience: 'general',
        topic: 'general',
        user_id: consistentUserId,
        card_type: 'black'
      };
      const blackResult = await cahApi.generateHumor(blackRequest);
      if (!blackResult.success || !blackResult.results || blackResult.results.length === 0) {
        throw new Error('Failed to generate black card');
      }
      const generatedBlackCard = blackResult.results[0];
      setBlackCard({
        id: generatedBlackCard.id,
        text: generatedBlackCard.text,
        pick: 1
      });
      
      // Generate white cards for this black card
      const whiteRequest: GenerateHumorRequest = {
        context: generatedBlackCard.text,
        audience: 'general',
        topic: 'general',
        user_id: consistentUserId,
        card_type: 'white'
      };
      const whiteResult = await cahApi.generateHumor(whiteRequest);
      if (whiteResult.success && whiteResult.results) {
        setBlackCardWhiteCards(whiteResult.results.slice(0, 3).map((g: any) => ({
          id: g.id,
          text: g.text,
          persona: g.persona_name
        })));
      }
      setRound(r => r + 1);
      toast.success('Black card and white cards generated!');
    } catch (error) {
      console.error('Black card generation error:', error);
      toast.error('Failed to generate black card game');
    } finally {
      setLoading(false);
    }
  };

  const fillSampleContext = () => {
    const randomContext = sampleContexts[Math.floor(Math.random() * sampleContexts.length)];
    setContext(randomContext);
  };

  // Multiplayer Game Functions
  const createGame = async () => {
    if (!user) {
      toast.error('Please log in to create a game');
      return;
    }

    try {
      setLoading(true);
      
      // Check authentication status
      const token = localStorage.getItem('cah_token');
      
      const result = await cahApi.createGame({
        max_score: 5,
        max_players: 6,
        round_timer: 300
      });
      


      if (result.success) {
        toast.success('Game created successfully!');
        setGameUserId(user.id);
        setIsInGame(true);
        setGameState(result.game_state);
        // Set the generated game ID
        setGameId(result.game_id);
        
        // Wait a moment for backend to fully initialize the game, then connect to WebSocket
        setTimeout(() => {
          connectToGame(result.game_id, user.id);
        }, 1000); // Wait 1 second for backend to finish game setup
      } else {
        console.error('Game creation failed:', result);
        toast.error(result.error || 'Failed to create game');
      }
    } catch (error: any) {
      console.error('Error creating game:', error);
      console.error('Error response:', error.response?.data);
      console.error('Error status:', error.response?.status);
      toast.error(`Failed to create game: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const joinGame = async () => {
    if (!gameId.trim()) {
      toast.error('Please enter a game ID');
      return;
    }

    if (!user) {
      toast.error('Please log in to join a game');
      return;
    }

    try {
      setLoading(true);
      // Check if token exists
      const token = localStorage.getItem('cah_token');
      
      const result = await cahApi.joinGame(gameId, user.id);


      
      if (result.success) {
        toast.success('Joined game successfully!');
        setGameState(result.game_state);
        setIsInGame(true);
        setGameUserId(user.id);
        // Connect to WebSocket
        connectToGame(gameId, user.id);
      } else {
        console.error('Join game failed:', result);
        toast.error(result.error || 'Failed to join game');
      }
    } catch (error: any) {
      console.error('Error joining game:', error);
      console.error('Error response:', error.response?.data);
      toast.error(`Failed to join game: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

    const connectToGame = (gameId: string, userId: string) => {

    
    // Add retry mechanism for WebSocket connection
    let retryCount = 0;
    const maxRetries = 3;
    
    const attemptConnection = () => {
      if (retryCount >= maxRetries) {
        console.error('❌ WebSocket connection failed after max retries');
        return;
      }
      
      // Use Render backend URL for WebSocket
      const wsUrl = process.env.NEXT_PUBLIC_API_URL?.replace('https://', 'wss://').replace('http://', 'ws://') || 'ws://localhost:8000';
      const ws = new WebSocket(`${wsUrl}/ws/${gameId}/${userId}`);
      
      // Set up periodic game state refresh for multiplayer games
      const refreshInterval = setInterval(() => {
        if (gameId && isInGame) {
    
          refreshGameState();
        }
      }, 10000); // Refresh every 10 seconds as fallback (reduced from 3 seconds)
      
      // Store the interval ID to clear it later
      setRefreshInterval(refreshInterval);
      
      ws.onopen = () => {

        setWebSocket(ws);
        
        // Send a join message to confirm connection
        ws.send(JSON.stringify({
          type: 'player_joined',
          game_id: gameId,
          user_id: userId
        }));
      };

      ws.onmessage = (event) => {
        const message = JSON.parse(event.data);

        
        // Handle different message types
        if (message.type === 'game_started') {
          
          if (message.game_state) {
            setGameState(message.game_state);

          } else {
            refreshGameState();
          }
          toast.success('Game started!');
        } else if (message.type === 'card_submitted') {
          // Update game state directly if provided, otherwise refresh
          if (message.game_state) {
            setGameState(message.game_state);
          } else {
            refreshGameState();
          }
        } else if (message.type === 'round_complete') {
          // Update game state directly if provided, otherwise refresh
          if (message.game_state) {
            setGameState(message.game_state);
          } else {
            refreshGameState();
          }
          toast.success(`Round complete! ${message.winning_card ? `Winner: ${message.winning_card}` : ''}`);
        } else if (message.type === 'player_joined') {
          // Immediately update the game state with the new player info
          if (message.game_state) {
            // Update the state immediately for instant UI update
            setGameState(message.game_state);
            
            // Show success message
            toast.success(`${message.new_player_username} joined the game!`);
          } else {
            // Fallback: refresh game state if we can't update immediately
            refreshGameState();
            toast.success(`${message.new_player_username || 'A player'} joined the game!`);
          }
        } else if (message.type === 'player_left') {
          // Update game state directly if provided, otherwise refresh
          if (message.game_state) {
            setGameState(message.game_state);
          } else {
            refreshGameState();
          }
          toast(`${message.username || 'A player'} left the game`);
        } else if (message.type === 'game_state_update') {
          // Handle general game state updates
          if (message.game_state) {
            setGameState(message.game_state);
          } else {
            refreshGameState();
          }
        } else if (message.type === 'game_state_updated') {
          // Refresh game state when cards are submitted
          refreshGameState();
          toast.success(message.message || 'Game state updated');
        } else if (message.type === 'phase_changed') {
          // Refresh game state when phase changes
          refreshGameState();
          toast.success(message.message || `Phase changed to ${message.new_phase}`);
        } else if (message.type === 'judge_ready_to_judge') {
          // Update game state with judge-specific data
          if (message.game_state) {
            setGameState(message.game_state);
          } else {
            refreshGameState();
          }
          toast.success(message.message || 'All cards submitted! You can now judge.');
        } else if (message.type === 'judge_ready_notification') {
          // Update game state for judge
          if (message.game_state) {
            setGameState(message.game_state);
          } else {
            refreshGameState();
          }
          toast.success(message.message || 'Ready to judge!');
        } else if (message.type === 'judge_waiting_notification') {
          // Update game state for judge during card submission
          if (message.game_state) {
            setGameState(message.game_state);
          } else {
            refreshGameState();
          }
          // Don't show toast for waiting notifications to avoid spam
        } else if (message.type === 'judge_submissions_update') {
          // Update game state when judge polls for submissions
          if (message.game_state) {
            setGameState(message.game_state);
          } else {
            refreshGameState();
          }
        } else if (message.type === 'judge_submissions_refresh') {
          // Handle judge-specific refresh
          if (message.game_state) {
            setGameState(message.game_state);
          } else {
            refreshGameState();
          }
        } else {
          // For any unknown message, refresh game state to be safe
          refreshGameState();
        }
      };

      ws.onclose = () => {
        setWebSocket(null);
        
        // When WebSocket closes, increase refresh frequency to compensate
        if (refreshInterval) {
          clearInterval(refreshInterval);
          const fallbackInterval = setInterval(() => {
            if (gameId && isInGame) {
              refreshGameState();
            }
          }, 5000); // Refresh every 5 seconds when WebSocket is down (reduced from 2 seconds)
          setRefreshInterval(fallbackInterval);
        }
        
        // Retry connection if it wasn't intentional
        retryCount++;
        setTimeout(attemptConnection, 2000); // Retry after 2 seconds
      };

      ws.onerror = (error) => {
        // On WebSocket error, refresh immediately
        refreshGameState();
        
        // Retry connection
        retryCount++;
        setTimeout(attemptConnection, 2000); // Retry after 2 seconds
      };
    };
    
    // Start the first connection attempt
    attemptConnection();
  };

  const refreshGameState = async () => {
    if (!gameId) {
      return;
    }
    
    try {
      const result = await cahApi.getGameState(gameId, gameUserId || userId);
      if (result.success) {
        const newPlayerCount = result.game_state?.players?.length || 0;
        const currentPlayerCount = gameState?.players?.length || 0;
        
        // If player count changed, this might be a player join/leave
        if (newPlayerCount !== currentPlayerCount) {
          // Schedule another refresh to ensure we catch any delayed updates
          setTimeout(() => {
            refreshGameState();
          }, 1500);
        }
        
        setGameState(result.game_state);
      }
    } catch (error) {
      console.error('❌ Error refreshing game state:', error);
    }
  };

  const startGame = async () => {
    if (!gameState || !gameState.players || !Array.isArray(gameState.players)) return;

    try {
      const hostPlayer = gameState.players.find(p => p.is_host);
      if (!hostPlayer) return;

      await cahApi.startGame(gameId, gameUserId || hostPlayer.user_id);
      toast.success('Game starting...');
    } catch (error) {
      console.error('Error starting game:', error);
      toast.error('Failed to start game');
    }
  };

  const submitCard = async (card: string) => {
    if (!gameState) return;

    try {
      // Only verify the card is in the current hand, don't refresh
      if (!gameState.my_hand?.includes(card)) {
        console.error('Card not found in current hand:', card);
        console.error('Available cards:', gameState.my_hand);
        toast.error('Card no longer available. Please select a different card.');
        // Refresh game state to get updated hand
        await refreshGameState();
        return;
      }
      
      await cahApi.submitCard(gameId, gameUserId || userId, card);
      toast.success('Card submitted!');
      // Refresh game state after successful submission
      await refreshGameState();
    } catch (error: any) {
      console.error('Error submitting card:', error);
      if (error.response?.status === 500 && error.response?.data?.detail === 'Card not in player\'s hand') {
        toast.error('Card no longer available. Please select a different card.');
        // Refresh game state to get updated hand
        await refreshGameState();
      } else if (error.response?.status === 500 && error.response?.data?.detail === 'Player has already submitted a card') {
        toast.error('You have already submitted a card for this round.');
        await refreshGameState();
      } else {
        toast.error('Failed to submit card');
        // Refresh game state to get updated hand
        await refreshGameState();
      }
    }
  };

  const judgeCard = async (winningCard: string) => {
    if (!gameState?.current_round) return;

    try {
      await cahApi.judgeRound(gameId, gameUserId || gameState.current_round.judge_id, winningCard);
      toast.success('Round judged!');
    } catch (error) {
      console.error('Error judging round:', error);
      toast.error('Failed to judge round');
    }
  };

  return (
    <div className="space-y-8">
      {/* Game Setup */}
      <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
        {user && (
          <div className="mb-6 p-4 bg-gradient-to-r from-green-500/20 to-blue-500/20 rounded-lg border border-green-500/30">
            <h3 className="text-lg font-semibold text-green-400 mb-2">
              🎉 Welcome back, {user.email}!
            </h3>
            <p className="text-gray-300 text-sm">
              Your cards will be personalized based on your learning history and preferences.
            </p>
          </div>
        )}
        <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
          <Play className="mr-2" />
          Game Setup
        </h2>

        {/* Card Type Selector */}
        <div className="mb-6">
          <label className="block text-white font-medium mb-2">Card Type</label>
          <div className="flex space-x-4">
            <button
              onClick={() => setCardType('multiplayer')}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                cardType === 'multiplayer'
                  ? 'bg-green-600 text-white shadow-lg'
                  : 'bg-white/20 text-white hover:bg-white/30'
              }`}
            >
              👥 Multiplayer
            </button>
            <button
              onClick={() => setCardType('white')}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                cardType === 'white'
                  ? 'bg-green-600 text-white shadow-lg'
                  : 'bg-white/20 text-white hover:bg-white/30'
              }`}
            >
              🃏 White Cards
            </button>
            <button
              onClick={() => setCardType('black')}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                cardType === 'black'
                  ? 'bg-green-600 text-white shadow-lg'
                  : 'bg-white/20 text-white hover:bg-white/30'
              }`}
            >
              ⚫ Black Cards
            </button>
            <button
              onClick={() => setCardType('game')}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                cardType === 'game'
                  ? 'bg-green-600 text-white shadow-lg'
                  : 'bg-white/20 text-white hover:bg-white/30'
              }`}
            >
              🎮 Game Mode
            </button>
          </div>
        </div>



        {/* Game Section - Only show when cardType is 'game' */}
        {cardType === 'game' && (
          <div className="mb-6">
            <h2 className="text-xl font-bold text-white mb-4 flex items-center">
              🖤 Black Cards Game
            </h2>
            
            <div className="mb-6">
              <label className="block text-white font-medium mb-2">Topic for Black Card</label>
              <textarea
                value={blackCardTopic}
                onChange={(e) => setBlackCardTopic(e.target.value)}
                placeholder="Enter a topic like: 'workplace humor' or 'gaming'"
                className="w-full p-4 rounded-lg bg-white/20 text-white placeholder-gray-300 border border-white/30 focus:border-purple-400 focus:outline-none resize-none"
                rows={2}
              />
            </div>
            
            <button
              onClick={handleBlackCardGenerate}
              disabled={loading || !blackCardTopic.trim()}
              className="w-full bg-gradient-to-r from-gray-800 to-black text-white font-bold py-4 px-6 rounded-lg hover:from-gray-700 hover:to-gray-900 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105 shadow-lg"
            >
              {loading ? (
                <div className="flex items-center justify-center">
                  <RotateCcw className="animate-spin mr-2" />
                  Generating...
                </div>
              ) : (
                <div className="flex items-center justify-center">
                  <Play className="mr-2" />
                  Generate Black Card
                </div>
              )}
            </button>
          </div>
        )}

        {/* Multiplayer Section - Only show when cardType is 'multiplayer' */}
        {cardType === 'multiplayer' && (
          <div className="mb-6">
            <h2 className="text-xl font-bold text-white mb-4 flex items-center">
              👥 Multiplayer Game
            </h2>
            
            {/* Prominent Game ID Display */}
            {gameId && (
              <div className="bg-gradient-to-r from-blue-500/20 to-purple-500/20 border border-blue-500/30 rounded-lg p-4 mb-4">
                <div className="text-center">
                  <div className="text-blue-400 font-medium mb-2">🎮 Active Game</div>
                  <div className="text-white text-sm mb-3">
                    Share this Game ID with other players:
                  </div>
                  <div className="bg-white/20 rounded-lg p-4 border border-blue-500/50">
                    <div className="text-3xl font-bold text-blue-400 font-mono tracking-widest mb-3">
                      {gameId}
                    </div>
                    <div className="flex gap-2 justify-center">
                      <button
                        onClick={() => {
                          navigator.clipboard.writeText(gameId);
                          toast.success('Game ID copied to clipboard!');
                        }}
                        className="px-4 py-2 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 transition-all"
                      >
                        📋 Copy Game ID
                      </button>
                      <button
                        onClick={() => {
                          // Share via Web Share API if available, fallback to clipboard
                          if (navigator.share) {
                            navigator.share({
                              title: 'Join my Cards Against Humanity game!',
                              text: `Join my Cards Against Humanity game! Use Game ID: ${gameId}`,
                            });
                          } else {
                            navigator.clipboard.writeText(`Join my Cards Against Humanity game! Use Game ID: ${gameId}`);
                            toast.success('Game invitation copied to clipboard!');
                          }
                        }}
                        className="px-4 py-2 bg-purple-600 text-white text-sm rounded hover:bg-purple-700 transition-all"
                      >
                        📤 Share Game
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
            

            {!isInGame ? (
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Create Game Section */}
                  <div className="bg-white/10 rounded-lg p-4">
                    <h3 className="text-lg font-bold text-white mb-3">🎮 Create New Game</h3>
                    <p className="text-gray-300 text-sm mb-4">
                      Start a new multiplayer game. A unique game ID will be generated automatically.
                    </p>
                    <button
                      onClick={createGame}
                      disabled={loading || !user}
                      className="w-full bg-gradient-to-r from-green-600 to-green-700 text-white font-bold py-3 px-6 rounded-lg hover:from-green-500 hover:to-green-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    >
                      {loading ? 'Creating...' : 'Create Game'}
                    </button>
                  </div>

                  {/* Join Game Section */}
                  <div className="bg-white/10 rounded-lg p-4">
                    <h3 className="text-lg font-bold text-white mb-3">🔗 Join Existing Game</h3>
                    <p className="text-gray-300 text-sm mb-4">
                      Enter a game ID to join an existing game.
                    </p>
                    <div className="mb-3">
                      <input
                        type="text"
                        value={gameId}
                        onChange={(e) => setGameId(e.target.value)}
                        placeholder="Enter game ID"
                        className="w-full p-3 rounded-lg bg-white/20 text-white placeholder-gray-300 border border-white/30 focus:border-green-400 focus:outline-none"
                      />
                    </div>
                    <button
                      onClick={joinGame}
                      disabled={loading || !user || !gameId.trim()}
                      className="w-full bg-gradient-to-r from-blue-600 to-blue-700 text-white font-bold py-3 px-6 rounded-lg hover:from-blue-500 hover:to-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    >
                      {loading ? 'Joining...' : 'Join Game'}
                    </button>
                  </div>
                </div>
              </div>
            ) : gameState ? (
              <div className="space-y-4">
                {/* Current Game Info */}
                <div className="bg-green-500/20 border border-green-500/30 rounded-lg p-4">
                  <div className="text-green-400 font-medium mb-2">🎮 Game in Progress</div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <div className="text-white text-sm mb-1">
                        <span className="font-semibold">Game ID:</span>
                      </div>
                      <div className="bg-white/20 rounded-lg p-2 border border-green-500/50">
                        <div className="text-lg font-bold text-green-400 font-mono tracking-wider text-center">
                          {gameId}
                        </div>
                      </div>
                      <button
                        onClick={() => {
                          navigator.clipboard.writeText(gameId);
                          toast.success('Game ID copied to clipboard!');
                        }}
                        className="mt-2 w-full px-3 py-1 bg-green-600 text-white text-xs rounded hover:bg-green-700 transition-all"
                      >
                        📋 Copy
                      </button>
                    </div>
                    <div>
                      <div className="text-white text-sm mb-1">
                        <span className="font-semibold">Status:</span> {gameState.status}
                      </div>
                      <div className="text-white text-sm mb-1">
                        <span className="font-semibold">Players:</span> {gameState.players?.length || 0}
                      </div>
                      <div className="text-white text-sm">
                        <span className="font-semibold">Host:</span> {gameState.players?.find(p => p.is_host)?.username || 'Unknown'}
                      </div>
                    </div>
                  </div>
                  
                  {/* Manual Refresh Button */}
                  <div className="mt-4 pt-4 border-t border-green-500/30">
                    <div className="text-center text-sm text-gray-300">
                      🔄 Real-time updates enabled - Game state updates automatically
                    </div>
                  </div>
                </div>


                
                {/* Game Status */}
                <div className="bg-white/10 rounded-lg p-4">
                  <h3 className="text-lg font-bold text-white mb-2">Game Status: {gameState.status.toUpperCase()}</h3>
                  <div className="text-gray-300">
                    Players: {gameState.players?.length || 0}/{gameState.max_players || 6} | 
                    Target Score: {gameState.max_score || 5}
                  </div>
                  
                  {/* Real-time status indicator */}
                  <div className="mt-3 pt-3 border-t border-white/20">
                    <div className="text-sm text-green-400 mb-2">
                      ✅ Real-time updates active
                    </div>
                    <div className="text-xs text-gray-400">
                      Players join/leave automatically - no refresh needed
                    </div>
                  </div>
                </div>

                {/* Players List */}
                <div className="bg-white/10 rounded-lg p-4">
                  <h3 className="text-lg font-bold text-white mb-2">Players ({gameState.players?.length || 0})</h3>
                  <div className="space-y-2">
                    {gameState.players && Array.isArray(gameState.players) ? gameState.players.map((player) => (
                      <div key={player.user_id} className="flex justify-between items-center text-white">
                        <span className="flex items-center">
                          {player.username}
                          {player.is_host && <span className="ml-2 text-yellow-400">👑</span>}
                          {player.is_judge && <span className="ml-2 text-blue-400">⚖️</span>}
                          {!player.connected && <span className="ml-2 text-red-400">🔴</span>}
                        </span>
                        <span className="font-bold">{player.score}</span>
                      </div>
                    )) : (
                      <div className="text-gray-400 text-center py-4">
                        No players data available
                      </div>
                    )}
                  </div>
                  
                  {/* Real-time player status */}
                  <div className="mt-3 pt-3 border-t border-white/20">
                    <div className="text-sm text-gray-300">
                                          {gameState.players?.length >= 2 
                      ? `✅ ${gameState.players.length} players ready - Host can start the game!`
                      : `⏳ Need ${2 - (gameState.players?.length || 0)} more player${(2 - (gameState.players?.length || 0)) === 1 ? '' : 's'} to start`
                    }
                    </div>
                  </div>
                </div>

                {/* Game Controls */}
                {gameState.status === 'waiting' && gameState.players && Array.isArray(gameState.players) && gameState.players.find(p => p.is_host)?.user_id === (gameUserId || userId) && (
                  <div className="space-y-2">

                    
                    <button
                      onClick={startGame}
                      disabled={gameState.players?.length < 2}
                      className="w-full bg-gradient-to-r from-green-600 to-green-700 text-white font-bold py-3 px-6 rounded-lg hover:from-green-500 hover:to-green-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    >
                      {gameState.players.length >= 2 
                        ? '🚀 Start Game!' 
                        : `Waiting for players... (${gameState.players?.length || 0}/2)`
                      }
                    </button>
                    
                    {/* Real-time status indicator */}
                    <div className="text-center text-sm text-gray-300">
                      {gameState.players?.length >= 2 
                        ? '✅ Ready to start! Click the button above to begin the game.'
                        : '⏳ Waiting for more players to join...'
                      }
                    </div>
                  </div>
                )}

                {/* Current Round */}
                {gameState.current_round && (
                  <div className="bg-white/10 rounded-lg p-4">
                    <h3 className="text-lg font-bold text-white mb-2">
                      Round {gameState.current_round.round_number} - {
                        gameState.current_round.judge_id === (gameUserId || userId) 
                          ? 'You are the Judge' 
                          : `Judge: ${gameState.current_round.judge_username}`
                      }
                    </h3>
                    <div className="bg-black text-white p-4 rounded-lg mb-4 text-center font-bold">
                      {gameState.current_round.black_card}
                    </div>

                    {/* Player's Hand - Only show to non-judge players */}
                    {gameState.current_round.phase === 'card_submission' && 
                     gameState.my_hand && 
                     gameState.my_hand.length > 0 && 
                     gameState.current_round?.judge_id !== (gameUserId || userId) && (
                      <div>
                        <h4 className="text-white font-medium mb-2">Your Cards:</h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                          {gameState.my_hand?.map((card, index) => (
                            <button
                              key={index}
                              onClick={() => submitCard(card)}
                              className="p-3 rounded-lg transition-all text-left bg-white text-black hover:bg-gray-100"
                            >
                              {card}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {/* Judge Message - Only show to judge */}
                    {gameState.current_round.phase === 'card_submission' && 
                     gameState.current_round?.judge_id === (gameUserId || userId) && (
                      <div className="text-center text-gray-300 py-4">
                        <p className="text-lg">⚖️ You are the Judge</p>
                        <p className="text-sm mt-2">Waiting for players to submit their white cards...</p>
                        <p className="text-xs mt-1">({gameState.current_round.submissions_count || 0} submitted)</p>
                        

                        
                        {/* Show submitted cards to judge during submission phase */}
                        {gameState.current_round.submissions && gameState.current_round.submissions.length > 0 && (
                          <div className="mt-4">
                            <h5 className="text-white font-medium mb-2">Submitted Cards:</h5>
                            <div className="space-y-2">
                              {gameState.current_round.submissions.map((submission, index) => (
                                <div key={index} className="bg-white/20 text-white p-3 rounded-lg border border-white/30">
                                  <span className="text-sm text-gray-300">{submission.player_username || `Player ${index + 1}`}:</span>
                                  <p className="text-white mt-1">{submission.card}</p>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Judging Phase */}
                    {gameState.current_round.phase === 'judging' && 
                     gameState.current_round.judge_id === (gameUserId || userId) && (
                      <div>
                        <h4 className="text-white font-medium mb-2">Choose the winner:</h4>
                        

                        
                        {gameState.current_round.submissions && gameState.current_round.submissions.length > 0 ? (
                          <div className="space-y-2">
                            {gameState.current_round.submissions.map((submission, index) => (
                              <button
                                key={index}
                                onClick={() => judgeCard(submission.card)}
                                className="w-full bg-white text-black p-3 rounded-lg hover:bg-gray-100 transition-all text-left"
                              >
                                <div className="text-left">
                                  <div className="text-sm text-gray-600 mb-1">
                                    {submission.player_username || `Player ${index + 1}`}
                                  </div>
                                  <div className="text-black font-medium">{submission.card}</div>
                                </div>
                              </button>
                            ))}
                          </div>
                        ) : (
                          <div className="text-center text-yellow-300 p-4 bg-yellow-500/20 rounded-lg border border-yellow-500/30">
                            ⚠️ No submissions available for judging. This might be a bug.
                            <div className="mt-2">
                              <button
                                onClick={() => {
                                  refreshGameState();
                                }}
                                className="px-3 py-1 bg-red-500 text-white text-xs rounded hover:bg-red-600 transition-colors"
                              >
                                🔄 Force Refresh
                              </button>
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Judge can see submissions during card submission phase */}
                    {gameState.current_round.phase === 'card_submission' && 
                     gameState.current_round.judge_id === (gameUserId || userId) && 
                     gameState.current_round.submissions && gameState.current_round.submissions.length > 0 && (
                      <div className="mb-4">
                        <h4 className="text-white font-medium mb-2">📝 Cards Submitted ({gameState.current_round.submissions.length}):</h4>
                        <div className="space-y-2">
                          {gameState.current_round.submissions.map((submission, index) => (
                            <div
                              key={index}
                              className="w-full bg-blue-500/20 text-white p-3 rounded-lg border border-blue-500/30"
                            >
                              <div className="text-left">
                                <div className="text-sm text-blue-300 mb-1">
                                  {submission.player_username || `Player ${index + 1}`}
                                </div>
                                <div className="text-white font-medium">{submission.card}</div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Waiting for submissions */}
                    {gameState.current_round.phase === 'card_submission' && !gameState.is_my_turn && (
                      <div className="text-center text-gray-300">
                        Waiting for players to submit cards... ({gameState.current_round.submissions_count || 0} submitted)
                      </div>
                    )}



                    {/* Judge waiting to judge */}
                    {gameState.current_round.phase === 'judging' && 
                     gameState.current_round.judge_id === (gameUserId || userId) && 
                     (!gameState.current_round.submissions || gameState.current_round.submissions.length === 0) && (
                      <div className="text-center text-gray-300">
                        Waiting for submissions to be ready for judging...
                      </div>
                    )}


                  </div>
                )}
              </div>
            ) : null}
          </div>
        )}

        {/* Context Input - Only show when cardType is not 'game' and not 'multiplayer' */}
        {cardType !== 'game' && cardType !== 'multiplayer' && (
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <label className="block text-white font-medium">
                {cardType === 'white' ? 'Black Card (with _____ for blank)' : 'Topic for Black Card'}
              </label>
              <button
                onClick={fillSampleContext}
                className="text-sm text-purple-300 hover:text-purple-200 flex items-center"
                title="Try a sample black card format"
              >
                <Sparkles className="w-4 h-4 mr-1" />
                Try Sample
              </button>
            </div>
            <textarea
              value={context}
              onChange={(e) => setContext(e.target.value)}
              placeholder={
                cardType === 'white' 
                  ? "Enter a black card like: 'TSA guidelines now prohibit _____ on airplanes.'"
                  : "Enter a topic like: 'workplace humor' or 'gaming'"
              }
              className="w-full p-4 rounded-lg bg-white/20 text-white placeholder-gray-300 border border-white/30 focus:border-purple-400 focus:outline-none resize-none"
              rows={3}
            />
            {cardType === 'white' && (
              <p className="text-xs text-gray-400 mt-1">
                💡 Enter a black card with _____ for the blank, or use &quot;Try Sample&quot; for examples
              </p>
            )}
          </div>
        )}

        {/* Settings Grid - Only show when not in multiplayer mode */}
        {cardType !== 'multiplayer' && (
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          {/* Audience Selection */}
          <div>
            <label className="block text-white font-medium mb-2">Audience</label>
            <div className="space-y-2">
              {audienceOptions.map((option) => (
                <button
                  key={option.value}
                  onClick={() => setAudience(option.value)}
                  className={`w-full p-3 rounded-lg text-left transition-all ${
                    audience === option.value
                      ? 'bg-purple-600 text-white shadow-lg'
                      : 'bg-white/20 text-white hover:bg-white/30'
                  }`}
                >
                  <div className="font-medium">{option.label}</div>
                  <div className="text-sm opacity-80">{option.description}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Topic Selection */}
          <div>
            <label className="block text-white font-medium mb-2">Topic</label>
            <div className="grid grid-cols-2 gap-2">
              {topicOptions.map((option) => (
                <button
                  key={option.value}
                  onClick={() => setTopic(option.value)}
                  className={`p-3 rounded-lg font-medium transition-all ${
                    topic === option.value
                      ? 'bg-indigo-600 text-white shadow-lg'
                      : 'bg-white/20 text-white hover:bg-white/30'
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
        </div>
        )}

        {/* Generate Button - Only show when not in multiplayer mode */}
        {cardType !== 'multiplayer' && (
        <button
          onClick={handleGenerate}
          disabled={loading || !context.trim()}
          className="w-full bg-gradient-to-r from-purple-600 to-indigo-600 text-white font-bold py-4 px-6 rounded-lg hover:from-purple-700 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105 shadow-lg"
        >
          {loading ? (
            <div className="flex items-center justify-center">
              <RotateCcw className="animate-spin mr-2" />
              Distributing...
            </div>
          ) : (
            <div className="flex items-center justify-center">
              <Play className="mr-2" />
              Distribute cards
            </div>
          )}
        </button>
        )}
      </div>

      {/* Recommendations */}
      {recommendations && recommendations.length > 0 && (
        <div className="bg-gradient-to-r from-purple-600/20 to-indigo-600/20 rounded-xl p-6 border border-purple-400/30">
          <h3 className="text-lg font-semibold text-white mb-3">
            🎭 Recommended AI Comedians for You
          </h3>
          <div className="flex flex-wrap gap-2">
            {recommendations.map((persona, index) => (
              <span
                key={index}
                className="px-3 py-1 bg-purple-600/40 text-purple-100 rounded-full text-sm font-medium"
              >
                {persona.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </span>
            ))}
          </div>
        </div>
      )}

      {context && (
        <div className="flex justify-center my-8">
          <div className="bg-black text-white rounded-2xl border-4 border-white shadow-2xl w-96 min-h-[120px] flex flex-col justify-center items-center p-8 font-bold text-2xl uppercase tracking-wide text-center select-none">
            {context}
          </div>
        </div>
      )}

      {/* Show black card and its white cards - Only when cardType is 'game' */}
      {cardType === 'game' && blackCard && (
        <div className="space-y-4 mt-8">
          <div className="flex items-center justify-center mb-4">
            <span className="text-lg font-bold text-white">Round {round}</span>
          </div>
          <div className="flex justify-center">
            <div className="bg-black text-white rounded-2xl border-4 border-white shadow-2xl w-96 min-h-[120px] flex flex-col justify-center items-center p-8 font-bold text-2xl uppercase tracking-wide text-center select-none">
              {blackCard.text}
            </div>
          </div>
          
          {blackCardWhiteCards && blackCardWhiteCards.length > 0 && (
            <div className="flex flex-row flex-wrap justify-center gap-6 mt-4">
              {blackCardWhiteCards.map((card) => (
                <div
                  key={card.id}
                  className="bg-white text-black border-4 border-black rounded-2xl shadow-xl w-80 min-h-[200px] flex flex-col justify-between items-center p-6 font-semibold text-base text-center select-none relative hover:scale-105 transition-transform"
                >
                  <div 
                    className="flex-grow flex items-center justify-center mb-4 cursor-pointer hover:bg-gray-50 rounded-lg p-2 transition-colors"
                    onClick={() => handleGameCardRating(card.id, 10)}
                  >
                    &quot;{card.text}&quot;
                  </div>
                  <div className="absolute top-2 right-2 text-xs bg-gray-200 text-gray-700 px-1.5 py-0.5 rounded-full font-bold shadow-sm">
                    {card.persona}
                  </div>
                  <div className="w-full flex flex-col items-center">
                    <div className="flex items-center space-x-1 mb-2">
                      <span className="px-1.5 py-0.5 bg-green-600/20 text-green-300 rounded text-xs font-medium">✅ Safe</span>
                      <span className="px-1.5 py-0.5 bg-blue-600/20 text-blue-300 rounded text-xs font-medium">
                        {gameCardRatings[card.id] || 5}/10
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-1 text-xs mb-2">
                      <div>
                        <span className="text-gray-400">Humor:</span>
                        <span className="text-black ml-1">8.5/10</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Creativity:</span>
                        <span className="text-black ml-1">8.2/10</span>
                      </div>
                      {/* Add Surprise Index for game mode */}
                      <div>
                        <span className="text-gray-400">Surprise:</span>
                        <span className="text-black ml-1">7.8/10</span>
                      </div>
                    </div>
                    <div className="w-full mb-2">
                      <label className="block text-xs text-gray-600 mb-1">Rate this card:</label>
                      <div className="flex justify-center space-x-1">
                        {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((rating) => (
                          <button
                            key={rating}
                            onClick={() => handleGameCardRating(card.id, rating)}
                            className={`w-6 h-6 rounded-full text-xs font-bold transition-all ${
                              gameCardRatings[card.id] === rating
                                ? 'bg-purple-600 text-white shadow-lg scale-110'
                                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                            }`}
                          >
                            {rating}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Results */}
      {generations && generations.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xl font-semibold text-white">
              
            </h3>
            <div className="text-sm text-gray-300">
              Generated in {generationTime.toFixed(2)}s
            </div>
          </div>

          {generations && generations.length > 0 && (
            <div className="flex items-center justify-center mb-2">
              <span className="text-lg font-bold text-white">Round {round}</span>
            </div>
          )}

          <div className="flex flex-row flex-wrap justify-center gap-6 mt-4">
            {generations.map((generation, index) => (
              <div
                key={generation.id}
                className={`${cardType === 'black' 
                  ? 'bg-black text-white border-4 border-white' 
                  : 'bg-white text-black border-4 border-black'
                } rounded-2xl shadow-xl w-72 min-h-[180px] flex flex-col justify-between items-center p-4 font-semibold text-base text-center select-none relative mb-4 hover:scale-105 transition-transform`}
              >
                <div 
                  className={`flex-grow flex items-center justify-center mb-2 ${cardType === 'white' ? 'cursor-pointer hover:bg-gray-50 rounded-lg p-2 transition-colors' : ''}`}
                  onClick={cardType === 'white' ? () => handleWhiteCardRating(generation, 10) : undefined}
                >
                  {cardType === 'black' ? generation.text.toUpperCase() : `"${generation.text}"`}
                </div>
                <div className={`absolute top-2 right-2 text-xs px-1.5 py-0.5 rounded-full font-bold shadow-sm ${
                  cardType === 'black' 
                    ? 'bg-white text-black' 
                    : 'bg-gray-200 text-gray-700'
                }`}>
                  {generation.persona_name.replace(/_/g, ' ')}
                </div>
                <div className="w-full flex flex-col items-center">
                  <div className="flex items-center space-x-1 mb-2">
                    <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${generation.is_safe ? 'bg-green-600/20 text-green-300' : 'bg-red-600/20 text-red-300'}`}>{generation.is_safe ? '✅ Safe' : '⚠️ Filtered'}</span>
                    <span className="px-1.5 py-0.5 bg-blue-600/20 text-blue-300 rounded text-xs font-medium">
                      {whiteCardRatings[generation.id] || 5}/10
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-1 text-xs mb-2">
                    <div>
                      <span className="text-gray-400">Humor:</span>
                      <span className={cardType === 'black' ? 'text-white ml-1' : 'text-black ml-1'}>
                        {generation.humor_score.toFixed(1)}/10
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-400">Creativity:</span>
                      <span className={cardType === 'black' ? 'text-white ml-1' : 'text-black ml-1'}>
                        {generation.creativity_score.toFixed(1)}/10
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-400">Appropriate:</span>
                      <span className={cardType === 'black' ? 'text-white ml-1' : 'text-black ml-1'}>
                        {generation.appropriateness_score.toFixed(1)}/10
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-400">Context:</span>
                      <span className={cardType === 'black' ? 'text-white ml-1' : 'text-black ml-1'}>
                        {generation.context_relevance_score?.toFixed(1) || 'N/A'}/10
                      </span>
                    </div>
                    {/* Add Surprise Index display */}
                    <div>
                      <span className="text-gray-400">Surprise:</span>
                      <span className={cardType === 'black' ? 'text-white ml-1' : 'text-black ml-1'}>
                        {generation.surprise_index?.toFixed(1) || 'N/A'}/10
                      </span>
                    </div>
                  </div>
                  {cardType === 'white' && (
                    <div className="w-full mb-2">
                      <label className="block text-xs text-gray-600 mb-1">Rate this card:</label>
                      <div className="flex justify-center space-x-1">
                        {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((rating) => (
                          <button
                            key={rating}
                            onClick={() => handleWhiteCardRating(generation, rating)}
                            className={`w-6 h-6 rounded-full text-xs font-bold transition-all ${
                              whiteCardRatings[generation.id] === rating
                                ? 'bg-purple-600 text-white shadow-lg scale-110'
                                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                            }`}
                          >
                            {rating}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  <div className="text-xs text-gray-500 mt-1 line-clamp-1">{generation.reasoning}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Empty State */}
      {!loading && (!generations || generations.length === 0) && (
        <div className="text-center py-12 text-gray-400">
          <Sparkles className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <h3 className="text-xl font-semibold mb-2">Ready to Generate Some Laughs?</h3>
          <p>Enter a context above and click &quot;Generate Humor&quot; to see AI-powered comedy in action!</p>
        </div>
      )}
    </div>
  );
}