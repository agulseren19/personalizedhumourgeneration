'use client';

import React, { useState, useEffect, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { Toaster } from 'react-hot-toast';
import GameInterface from '../Components/cah/GameInterface';
import Analytics from '../Components/cah/Analytics';
import PersonaShowcase from '../Components/cah/PersonaShowcase';
import { generateUserId } from '../lib/api';
import { useUser } from '../contexts/UserContext';
import { Gamepad2, BarChart3, Users, Gamepad, CheckCircle, Rocket, Brain, Shield, Cloud } from 'lucide-react';

function CAHPageContent() {
  const { user, loading } = useUser();
  const [userId, setUserId] = useState<string>('');
  const [activeTab, setActiveTab] = useState<'game' | 'analytics' | 'personas'>('game');
  const searchParams = useSearchParams();

  useEffect(() => {
    if (user) {
      // Clear guest userId when user is authenticated
      setUserId('');
    } else {
      // Only generate a guest userId if user is not authenticated
      setUserId(generateUserId());
    }
  }, [user]);

  useEffect(() => {
    // Handle URL query parameters for tab selection
    const tabParam = searchParams.get('tab');
    if (tabParam === 'analytics') {
      setActiveTab('analytics');
    } else if (tabParam === 'personas') {
      setActiveTab('personas');
    } else {
      setActiveTab('game');
    }
  }, [searchParams]);

  const tabs = [
    { id: 'game', label: 'Play Game', icon: <Gamepad2 size={20} /> },
    { id: 'analytics', label: 'Your Learning', icon: <BarChart3 size={20} /> },
    { id: 'personas', label: 'AI Comedians', icon: <Users size={20} /> }
  ];

  return (
    <div className="min-h-screen bg-background-seaGreen pt-32">
      <Toaster position="top-right" />
      
      {/* Header */}
      <div className="bg-black/20 backdrop-blur-sm border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-white mb-2">
                <Gamepad className="inline mr-2" size={24} />
                AI Cards Against Humanity
              </h1>
              <p className="text-gray-300">
                AI-powered humor generation with personalized learning
              </p>
            </div>
            <div className="text-sm text-gray-400">
              {user ? (
                <div className="flex items-center space-x-3">
                  <div className="bg-green-500/20 border border-green-500/50 rounded-full px-3 py-1 text-green-400">
                    <CheckCircle className="inline mr-2" size={16} />
                    Logged in as {user.email}
                  </div>
                  <button 
                    onClick={() => {
                      localStorage.removeItem('cah_token');
                      localStorage.removeItem('user_info');
                      window.location.reload();
                    }}
                    className="bg-red-500/20 border border-red-500/50 rounded-full px-3 py-1 text-red-400 hover:bg-red-500/30 transition-colors"
                  >
                    Logout
                  </button>
                </div>
              ) : (
                <div className="text-gray-400">
                  Guest User ID: {userId ? userId.slice(-8) : 'None'}
                </div>
              )}
            </div>
          </div>
          
          {/* Navigation Tabs */}
          <div className="flex space-x-1 mt-6">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`
                  px-6 py-3 rounded-lg font-medium transition-all duration-200 flex items-center space-x-2
                                      ${activeTab === tab.id
                     ? 'bg-accent-green text-white shadow-lg'
                     : 'bg-white/10 text-white hover:bg-accent-green/20'
                    }
                `}
              >
{tab.icon}
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-4 py-12">
        {activeTab === 'game' && <GameInterface userId={user?.id || userId} />}
        {activeTab === 'analytics' && <Analytics userId={user?.id || userId} />}
        {activeTab === 'personas' && <PersonaShowcase />}
      </div>
      
      {/* Debug Info - Remove in production */}
      {process.env.NODE_ENV === 'development' && (
        <div className="fixed bottom-4 right-4 bg-black/80 text-white p-4 rounded-lg text-sm max-w-xs">
          <div className="font-bold mb-2">🔍 Debug Info</div>
          <div>Authenticated User: {user ? 'Yes' : 'No'}</div>
          <div>User ID: {user?.id || 'None'}</div>
          <div>Guest ID: {userId || 'None'}</div>
          <div>Effective ID: {user?.id || userId || 'None'}</div>
          <div>User Email: {user?.email || 'None'}</div>
          <div className="mt-2">
            <button 
              onClick={() => window.location.reload()}
              className="bg-blue-500 hover:bg-blue-600 px-2 py-1 rounded text-xs"
            >
              🔄 Refresh Page
            </button>
            <button 
              onClick={() => {/* console.log('🔍 Current State:', { user, userId, effectiveId: user?.id || userId }) */}}
              className="bg-green-500 hover:bg-green-600 px-2 py-1 rounded text-xs ml-1"
            >
              <BarChart3 className="inline mr-2" size={16} />
              Log State
            </button>
          </div>
        </div>
      )}

      {/* Features Banner */}
      <div className="bg-background-darkBlue/30 border-t border-white/10 mt-12">
        <div className="max-w-7xl mx-auto px-4 py-8">
          <h3 className="text-2xl font-bold text-white mb-6 text-center">
            <Rocket className="inline mr-2" size={20} />
            Powered by Advanced AI Features
          </h3>
          <div className="grid md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="mb-2">
                <Brain size={36} className="mx-auto text-accent-blue" />
              </div>
              <h4 className="font-semibold text-white mb-1">Smart Learning</h4>
              <p className="text-sm text-gray-300">Learns your humor preferences over time</p>
            </div>
            <div className="text-center">
              <div className="mb-2">
                <Users size={36} className="mx-auto text-accent-orange" />
              </div>
              <h4 className="font-semibold text-white mb-1">AI Personas</h4>
              <p className="text-sm text-gray-300">Multiple comedy personalities and styles</p>
            </div>
            <div className="text-center">
              <div className="mb-2">
                <Shield size={36} className="mx-auto text-accent-yellow" />
              </div>
              <h4 className="font-semibold text-white mb-1">Content Filter</h4>
              <p className="text-sm text-gray-300">Detoxify-powered safety checking</p>
            </div>
            <div className="text-center">
              <div className="mb-2">
                <Cloud size={36} className="mx-auto text-custom-brown2" />
              </div>
              <h4 className="font-semibold text-white mb-1">CrewAI Orchestration</h4>
              <p className="text-sm text-gray-300">Multi-agent AI collaboration and orchestration</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function CAHPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <CAHPageContent />
    </Suspense>
  );
} 