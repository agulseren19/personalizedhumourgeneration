'use client';

import React, { useState, useEffect } from 'react';
import { cahApi, UserAnalytics } from '../../lib/api';
import { TrendingUp, TrendingDown, Brain, BarChart3 } from 'lucide-react';
import { useUser } from '../../contexts/UserContext';

interface AnalyticsProps {
  userId: string;
}

export default function Analytics({ userId }: AnalyticsProps) {
  const { user } = useUser();
  const [analytics, setAnalytics] = useState<UserAnalytics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const effectiveUserId = user?.id || userId;
    
    // Debug logging
    console.log('🔍 Analytics Debug:', {
      authenticatedUserId: user?.id,
      fallbackUserId: userId,
      effectiveUserId
    });
    
    if (effectiveUserId) {
      loadAnalytics(effectiveUserId);
    }
  }, [user, userId]);

  const loadAnalytics = async (effectiveUserId: string) => {
    try {
      setLoading(true);
      const result = await cahApi.getUserAnalytics(effectiveUserId);
      
      if (result.success) {
        setAnalytics(result.analytics);
        setError(null);
      } else {
        setError('Failed to load analytics');
      }
    } catch (err) {
      setError('Error loading analytics data');
      console.error('Analytics error:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-400"></div>
        <span className="ml-3 text-white">Loading your learning data...</span>
      </div>
    );
  }

  if (error || !analytics) {
    return (
      <div className="bg-white/10 backdrop-blur-sm rounded-xl p-8 border border-white/20 text-center">
        <Brain className="w-16 h-16 mx-auto mb-4 text-gray-400" />
        <h3 className="text-xl font-semibold text-white mb-2">No Learning Data Yet</h3>
        <p className="text-gray-300 mb-4">
          Start playing the game to see your personalized learning analytics!
        </p>
        <button
          onClick={() => {
            const effectiveUserId = user?.id || userId;
            if (effectiveUserId) {
              loadAnalytics(effectiveUserId);
            }
          }}
          className="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 transition-colors"
        >
          Refresh
        </button>
      </div>
    );
  }

  const getPersonaStatusColor = (status: string) => {
    switch (status) {
      case 'liked':
        return 'text-green-400 bg-green-400/20';
      case 'disliked':
        return 'text-red-400 bg-red-400/20';
      default:
        return 'text-gray-400 bg-gray-400/20';
    }
  };

  const getPersonaStatusEmoji = (status: string) => {
    switch (status) {
      case 'liked':
        return '💚';
      case 'disliked':
        return '💔';
      default:
        return '🤍';
    }
  };

  return (
    <div className="space-y-8">
      {/* Personalized Welcome */}
      {user && (
        <div className="bg-gradient-to-r from-green-500/20 to-blue-500/20 rounded-xl p-6 border border-green-500/30">
          <h2 className="text-2xl font-bold text-green-400 mb-2">
            📊 Your Learning Analytics
          </h2>
          <p className="text-gray-300">
            Personalized insights for {user.email} - See how the AI learns your humor preferences!
          </p>
        </div>
      )}
      
      {/* Overview Cards */}
      <div className="grid md:grid-cols-3 gap-6">
        <div className="bg-gradient-to-br from-purple-600/20 to-purple-800/20 rounded-xl p-6 border border-purple-400/30">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">Total Interactions</h3>
            <BarChart3 className="w-6 h-6 text-purple-400" />
          </div>
          <div className="text-3xl font-bold text-white">{analytics.total_interactions}</div>
          <div className="text-sm text-purple-300 mt-2">
            Times you&apos;ve played and given feedback
          </div>
        </div>

        <div className="bg-gradient-to-br from-indigo-600/20 to-indigo-800/20 rounded-xl p-6 border border-indigo-400/30">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">Average Score</h3>
            <TrendingUp className="w-6 h-6 text-indigo-400" />
          </div>
          <div className="text-3xl font-bold text-white">{analytics.average_score.toFixed(1)}/10</div>
          <div className="text-sm text-indigo-300 mt-2">
            Your overall satisfaction rating
          </div>
        </div>

        <div className="bg-gradient-to-br from-green-600/20 to-green-800/20 rounded-xl p-6 border border-green-400/30">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">Learning Progress</h3>
            <Brain className="w-6 h-6 text-green-400" />
          </div>
          <div className="text-3xl font-bold text-white">
            {analytics.liked_personas.length + analytics.disliked_personas.length}
          </div>
          <div className="text-sm text-green-300 mt-2">
            Personas with established preferences
          </div>
        </div>
      </div>

      {/* Liked vs Disliked Personas */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Liked Personas */}
        <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
          <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
            <span className="text-green-400 mr-2">💚</span>
            Your Favorite AI Comedians ({analytics.liked_personas.length})
          </h3>
          {analytics.liked_personas.length > 0 ? (
            <div className="space-y-3">
              {analytics.liked_personas.map((persona) => (
                <div
                  key={persona}
                  className="flex items-center justify-between p-3 bg-green-600/20 rounded-lg"
                >
                  <span className="text-white font-medium">
                    {persona.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </span>
                  <div className="flex items-center space-x-2">
                    <span className="text-green-300 text-sm">
                      {analytics.persona_performance[persona]?.avg_score.toFixed(1)}/10
                    </span>
                    <span className="text-green-400">
                      ({analytics.persona_performance[persona]?.interaction_count} times)
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-400">
              <div className="text-4xl mb-2">🎭</div>
              <p>No favorite comedians yet!</p>
              <p className="text-sm">Rate responses 7+ to build your preferences</p>
            </div>
          )}
        </div>

        {/* Disliked Personas */}
        <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
          <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
            <span className="text-red-400 mr-2">💔</span>
            AI Comedians You Don&apos;t Prefer ({analytics.disliked_personas.length})
          </h3>
          {analytics.disliked_personas.length > 0 ? (
            <div className="space-y-3">
              {analytics.disliked_personas.map((persona) => (
                <div
                  key={persona}
                  className="flex items-center justify-between p-3 bg-red-600/20 rounded-lg"
                >
                  <span className="text-white font-medium">
                    {persona.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </span>
                  <div className="flex items-center space-x-2">
                    <span className="text-red-300 text-sm">
                      {analytics.persona_performance[persona]?.avg_score.toFixed(1)}/10
                    </span>
                    <span className="text-red-400">
                      ({analytics.persona_performance[persona]?.interaction_count} times)
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-400">
              <div className="text-4xl mb-2">👍</div>
              <p>No dislikes yet!</p>
              <p className="text-sm">The system will learn what you don&apos;t like</p>
            </div>
          )}
        </div>
      </div>

      {/* Detailed Persona Performance */}
      <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20">
        <h3 className="text-xl font-semibold text-white mb-6 flex items-center">
          <BarChart3 className="mr-2" />
          Detailed Persona Performance
        </h3>
        
        <div className="grid gap-4">
          {Object.entries(analytics.persona_performance)
            .sort(([,a], [,b]) => b.avg_score - a.avg_score)
            .map(([persona, data]) => (
              <div
                key={persona}
                className="flex items-center justify-between p-4 bg-white/5 rounded-lg hover:bg-white/10 transition-all"
              >
                <div className="flex items-center space-x-4">
                  <span className={`text-2xl`}>
                    {getPersonaStatusEmoji(data.status)}
                  </span>
                  <div>
                    <div className="text-white font-medium">
                      {persona.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </div>
                    <div className="text-sm text-gray-400">
                      {data.interaction_count} interaction{data.interaction_count !== 1 ? 's' : ''}
                    </div>
                  </div>
                </div>

                <div className="flex items-center space-x-4">
                  <div className="text-right">
                    <div className="text-white font-bold">
                      {data.avg_score.toFixed(1)}/10
                    </div>
                    <div className={`text-xs px-2 py-1 rounded-full ${getPersonaStatusColor(data.status)}`}>
                      {data.status}
                    </div>
                  </div>
                  
                  {/* Score Bar */}
                  <div className="w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all duration-500 ${
                        data.avg_score >= 7
                          ? 'bg-green-500'
                          : data.avg_score <= 4
                          ? 'bg-red-500'
                          : 'bg-yellow-500'
                      }`}
                      style={{ width: `${(data.avg_score / 10) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
        </div>
        
        {Object.keys(analytics.persona_performance).length === 0 && (
          <div className="text-center py-8 text-gray-400">
            <BarChart3 className="w-16 h-16 mx-auto mb-4 opacity-50" />
            <h4 className="text-lg font-semibold mb-2">No Performance Data</h4>
            <p>Start playing and rating responses to see detailed analytics!</p>
          </div>
        )}
      </div>

      {/* Learning Insights */}
      <div className="bg-gradient-to-r from-purple-600/20 to-indigo-600/20 rounded-xl p-6 border border-purple-400/30">
        <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
          <Brain className="mr-2" />
          Your Learning Insights
        </h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-semibold text-purple-300 mb-2">What the AI Has Learned About You:</h4>
            <ul className="space-y-2 text-sm text-white">
              {analytics.liked_personas.length > 0 && (
                <li className="flex items-center">
                  <span className="text-green-400 mr-2">✓</span>
                  You prefer {analytics.liked_personas[0]?.replace(/_/g, ' ')} style comedy
                </li>
              )}
              {analytics.average_score > 6 && (
                <li className="flex items-center">
                  <span className="text-green-400 mr-2">✓</span>
                  You generally enjoy the AI-generated humor
                </li>
              )}
              {analytics.disliked_personas.length > 0 && (
                <li className="flex items-center">
                  <span className="text-red-400 mr-2">✗</span>
                  You avoid {analytics.disliked_personas[0]?.replace(/_/g, ' ')} style comedy
                </li>
              )}
              {analytics.total_interactions > 5 && (
                <li className="flex items-center">
                  <span className="text-blue-400 mr-2">→</span>
                  You&apos;re an active user building strong preferences
                </li>
              )}
            </ul>
          </div>
          
          <div>
            <h4 className="font-semibold text-indigo-300 mb-2">How This Improves Your Experience:</h4>
            <ul className="space-y-2 text-sm text-white">
              <li className="flex items-center">
                <span className="text-purple-400 mr-2">🎯</span>
                Personas are ranked by your preferences
              </li>
              <li className="flex items-center">
                <span className="text-purple-400 mr-2">🚫</span>
                Disliked personas are filtered out
              </li>
              <li className="flex items-center">
                <span className="text-purple-400 mr-2">📈</span>
                Recommendations get more accurate over time
              </li>
              <li className="flex items-center">
                <span className="text-purple-400 mr-2">🧠</span>
                AWS Knowledge Base stores your patterns
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
} 