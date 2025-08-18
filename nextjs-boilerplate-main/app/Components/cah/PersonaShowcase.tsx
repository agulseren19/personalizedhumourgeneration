'use client';

import React, { useState, useEffect } from 'react';
import { cahApi, Persona } from '../../lib/api';
import { Users, Sparkles, Brain, Zap } from 'lucide-react';

export default function PersonaShowcase() {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedPersona, setSelectedPersona] = useState<Persona | null>(null);

  useEffect(() => {
    loadPersonas();
  }, []);

  const loadPersonas = async () => {
    try {
      setLoading(true);
      const result = await cahApi.getPersonas();
      
      if (result.success) {
        setPersonas(result.personas);
      }
    } catch (error) {
      console.error('Error loading personas:', error);
    } finally {
      setLoading(false);
    }
  };

  const getPersonaEmoji = (personaId: string) => {
    const emojiMap: Record<string, string> = {
      'dad_humor_enthusiast': '👨‍👧‍👦',
      'millennial_memer': '😂',
      'office_worker': '💼',
      'gaming_guru': '🎮',
      'dark_humor_specialist': '🌚',
      'suburban_parent': '🏠',
      'gen_z_chaos': '🔥',
      'wordplay_master': '🎭',
      'corporate_ladder_climber': '📈',
      'absurdist_artist': '🎨'
    };
    return emojiMap[personaId] || '🎭';
  };

  const getPersonaGradient = (index: number) => {
    const gradients = [
      'from-purple-500 to-pink-500',
      'from-blue-500 to-cyan-500',
      'from-green-500 to-emerald-500',
      'from-yellow-500 to-orange-500',
      'from-red-500 to-pink-500',
      'from-indigo-500 to-purple-500',
      'from-teal-500 to-green-500',
      'from-orange-500 to-red-500'
    ];
    return gradients[index % gradients.length];
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-400"></div>
        <span className="ml-3 text-white">Loading AI comedians...</span>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-bold text-white mb-4 flex items-center justify-center">
          <Users className="mr-3" />
          Meet Your AI Comedy Squad
        </h2>
        <p className="text-gray-300 max-w-2xl mx-auto">
          Each AI comedian has their own personality, humor style, and expertise. 
          The system learns which ones you prefer and recommends them for future generations.
        </p>
      </div>

      {/* Personas Grid */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        {personas.map((persona, index) => (
          <div
            key={persona.id}
            onClick={() => setSelectedPersona(persona)}
            className="group cursor-pointer transform hover:scale-105 transition-all duration-300"
          >
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20 hover:border-white/40 hover:bg-white/20 transition-all">
              {/* Persona Header */}
              <div className="flex items-center mb-4">
                <div className={`text-4xl mr-4 p-3 rounded-full bg-gradient-to-r ${getPersonaGradient(index)} bg-opacity-20`}>
                  {getPersonaEmoji(persona.id)}
                </div>
                <div>
                  <h3 className="text-lg font-bold text-white group-hover:text-purple-300 transition-colors">
                    {persona.name}
                  </h3>
                  <div className="text-sm text-purple-300 font-medium">
                    {persona.humor_style}
                  </div>
                </div>
              </div>

              {/* Description */}
              <p className="text-gray-300 text-sm mb-4 line-clamp-3">
                {persona.description}
              </p>

              {/* Expertise Tags */}
              <div className="flex flex-wrap gap-2 mb-4">
                {persona.expertise.slice(0, 3).map((expertise, idx) => (
                  <span
                    key={idx}
                    className="px-2 py-1 bg-purple-600/30 text-purple-200 rounded-full text-xs font-medium"
                  >
                    {expertise}
                  </span>
                ))}
                {persona.expertise.length > 3 && (
                  <span className="px-2 py-1 bg-gray-600/30 text-gray-300 rounded-full text-xs">
                    +{persona.expertise.length - 3} more
                  </span>
                )}
              </div>

              {/* Click to view more */}
              <div className="text-center pt-2 border-t border-white/10">
                <span className="text-xs text-gray-400 group-hover:text-purple-300 transition-colors">
                  Click to learn more
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* System Features */}
      <div className="bg-gradient-to-r from-purple-600/20 to-indigo-600/20 rounded-xl p-8 border border-purple-400/30">
        <h3 className="text-2xl font-bold text-white mb-6 text-center">
          🧠 How the AI Comedy System Works
        </h3>
        
        <div className="grid md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="bg-purple-600/20 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
              <Brain className="w-8 h-8 text-purple-400" />
            </div>
            <h4 className="font-semibold text-white mb-2">Smart Learning</h4>
            <p className="text-sm text-gray-300">
              The system learns from your feedback ratings (1-10) to understand your humor preferences over time.
            </p>
          </div>
          
          <div className="text-center">
            <div className="bg-indigo-600/20 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
              <Sparkles className="w-8 h-8 text-indigo-400" />
            </div>
            <h4 className="font-semibold text-white mb-2">Personalized Recommendations</h4>
            <p className="text-sm text-gray-300">
              Personas you rate highly (7+) are prioritized, while those you dislike (4-) are filtered out.
            </p>
          </div>
          
          <div className="text-center">
            <div className="bg-green-600/20 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
              <Zap className="w-8 h-8 text-green-400" />
            </div>
            <h4 className="font-semibold text-white mb-2">Multiple AI Models</h4>
            <p className="text-sm text-gray-300">
              Each persona can use different AI models (GPT-4, Claude, DeepSeek) for diverse humor generation.
            </p>
          </div>
        </div>
      </div>

      {/* Selected Persona Modal */}
      {selectedPersona && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 z-50">
          <div className="bg-gray-900 rounded-xl p-8 max-w-2xl w-full max-h-[80vh] overflow-y-auto border border-white/20">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center">
                <div className="text-5xl mr-4">
                  {getPersonaEmoji(selectedPersona.id)}
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-white">
                    {selectedPersona.name}
                  </h2>
                  <div className="text-purple-300 font-medium">
                    {selectedPersona.humor_style}
                  </div>
                </div>
              </div>
              <button
                onClick={() => setSelectedPersona(null)}
                className="text-gray-400 hover:text-white text-2xl"
              >
                ×
              </button>
            </div>

            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-white mb-2">About This Comedian</h3>
                <p className="text-gray-300">
                  {selectedPersona.description}
                </p>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-white mb-3">Areas of Expertise</h3>
                <div className="flex flex-wrap gap-2">
                  {selectedPersona.expertise.map((expertise, idx) => (
                    <span
                      key={idx}
                      className="px-3 py-1 bg-purple-600/30 text-purple-200 rounded-full text-sm font-medium"
                    >
                      {expertise}
                    </span>
                  ))}
                </div>
              </div>

              <div className="bg-purple-600/20 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-white mb-2">How to Interact</h3>
                <ul className="text-sm text-gray-300 space-y-1">
                  <li>• This persona will be recommended based on context and your preferences</li>
                  <li>• Rate their responses 1-10 to help the system learn your taste</li>
                  <li>• High ratings (7+) make them appear more often</li>
                  <li>• Low ratings (4-) reduce their future appearances</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
} 