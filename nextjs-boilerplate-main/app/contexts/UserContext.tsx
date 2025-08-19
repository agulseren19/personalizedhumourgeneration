'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface User {
  id: string;
  email: string;
  username?: string;
  created_at: string;
  preferences?: {
    favorite_personas: string[];
    humor_style: string;
    audience_preference: string;
  };
}

interface UserContextType {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<boolean>;
  register: (email: string, password: string) => Promise<boolean>;
  logout: () => void;
  updateUserPreferences: (preferences: Partial<User['preferences']>) => Promise<void>;
  refreshAuth: () => void;
  debugTokenStatus: () => {
    tokenExists: boolean;
    tokenLength: number;
    tokenValue: string | null;
    userInfoExists: boolean;
    currentUser: User | null;
  };
  setTokenManually: (token: string) => void;
  clearTokenAndReload: () => void;
  validateCurrentToken: () => Promise<boolean>;
}

const UserContext = createContext<UserContextType | undefined>(undefined);

export function UserProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  // Check for existing session on mount
  useEffect(() => {
    checkAuthStatus();
    
    // Listen for auth success events (from Google OAuth)
    const handleAuthSuccess = () => {
      console.log('🔍 Auth success event received, refreshing auth status...');
      checkAuthStatus();
    };
    
    window.addEventListener('auth-success', handleAuthSuccess);
    
    return () => {
      window.removeEventListener('auth-success', handleAuthSuccess);
    };
  }, []);

  const checkAuthStatus = async () => {
    try {
      console.log('🔍 Checking auth status...');
      const token = localStorage.getItem('cah_token');
      console.log('🔍 Token found:', !!token);
      console.log('🔍 Token preview:', token ? `${token.substring(0, 20)}...` : 'None');
      console.log('🔍 Token length:', token ? token.length : 0);
      console.log('🔍 Token value:', token);
      
      if (token && token.trim() !== '') {
        try {
          const response = await fetch('/api/auth/me', {
            headers: { Authorization: `Bearer ${token}` }
          });
          
          if (response.ok) {
            const userData = await response.json();
            console.log('🔍 User data from API:', userData);
            setUser(userData.user);
          } else {
            console.log('🔍 Token invalid, removing...');
            localStorage.removeItem('cah_token');
            localStorage.removeItem('user_info');
            setUser(null);
          }
        } catch (error) {
          console.error('🔍 API call failed:', error);
          // Don't remove token on network errors, just set user to null
          setUser(null);
        }
      } else {
        console.log('🔍 No valid token found, checking user_info...');
        // Check if we have user_info from Google OAuth
        const userInfo = localStorage.getItem('user_info');
        console.log('🔍 User info found:', !!userInfo);
        if (userInfo) {
          try {
            const user = JSON.parse(userInfo);
            console.log('🔍 User from user_info:', user);
            setUser(user);
          } catch (e) {
            console.log('🔍 Failed to parse user_info:', e);
            localStorage.removeItem('user_info');
            setUser(null);
          }
        } else {
          setUser(null);
        }
      }
    } catch (error) {
      console.error('Auth check failed:', error);
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  const login = async (email: string, password: string): Promise<boolean> => {
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });

      const data = await response.json();

      if (data.success) {
        console.log('🔍 Login successful, storing token:', {
          success: data.success,
          hasAccessToken: !!data.access_token,
          accessTokenPreview: data.access_token ? `${data.access_token.substring(0, 20)}...` : 'None',
          user: data.user,
          fullResponse: data
        });
        
        // Check if access_token exists and is not empty
        if (!data.access_token || data.access_token.trim() === '') {
          console.error('🔍 ERROR: access_token is missing or empty!');
          console.error('🔍 Full response data:', data);
          return false;
        }
        
        localStorage.setItem('cah_token', data.access_token);
        setUser(data.user);
        
        // Verify token is stored correctly
        const storedToken = localStorage.getItem('cah_token');
        console.log('🔍 Token stored successfully:', !!storedToken);
        console.log('🔍 Stored token preview:', storedToken ? `${storedToken.substring(0, 20)}...` : 'None');
        console.log('🔍 Stored token length:', storedToken ? storedToken.length : 0);
        
        return true;
      } else {
        throw new Error(data.error || 'Login failed');
      }
    } catch (error) {
      console.error('Login error:', error);
      return false;
    }
  };

  const register = async (email: string, password: string): Promise<boolean> => {
    try {
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });

      const data = await response.json();

      if (data.success) {
        console.log('🔍 Registration successful, storing token:', {
          success: data.success,
          hasAccessToken: !!data.access_token,
          accessTokenPreview: data.access_token ? `${data.access_token.substring(0, 20)}...` : 'None',
          user: data.user,
          fullResponse: data
        });
        
        // Check if access_token exists and is not empty
        if (!data.access_token || data.access_token.trim() === '') {
          console.error('🔍 ERROR: access_token is missing or empty!');
          console.error('🔍 Full response data:', data);
          return false;
        }
        
        localStorage.setItem('cah_token', data.access_token);
        setUser(data.user);
        
        // Verify token is stored correctly
        const storedToken = localStorage.getItem('cah_token');
        console.log('🔍 Token stored successfully:', !!storedToken);
        console.log('🔍 Stored token preview:', storedToken ? `${storedToken.substring(0, 20)}...` : 'None');
        console.log('🔍 Stored token length:', storedToken ? storedToken.length : 0);
        
        return true;
      } else {
        throw new Error(data.error || 'Registration failed');
      }
    } catch (error) {
      console.error('Registration error:', error);
      return false;
    }
  };

  const logout = () => {
    localStorage.removeItem('cah_token');
    localStorage.removeItem('user_info');
    setUser(null);
  };

  const refreshAuth = () => {
    checkAuthStatus();
  };

  // Debug function to check token status
  const debugTokenStatus = () => {
    const token = localStorage.getItem('cah_token');
    const userInfo = localStorage.getItem('user_info');
    
    console.log('🔍 DEBUG TOKEN STATUS:');
    console.log('🔍 Token exists:', !!token);
    console.log('🔍 Token length:', token ? token.length : 0);
    console.log('🔍 Token value:', token);
    console.log('🔍 Token trimmed:', token ? token.trim() : 'N/A');
    console.log('🔍 User info exists:', !!userInfo);
    console.log('🔍 User info:', userInfo);
    console.log('🔍 Current user state:', user);
    
    return {
      tokenExists: !!token,
      tokenLength: token ? token.length : 0,
      tokenValue: token,
      userInfoExists: !!userInfo,
      currentUser: user
    };
  };

  // Debug function to manually set token
  const setTokenManually = (token: string) => {
    console.log('🔍 Manually setting token:', token.substring(0, 20) + '...');
    localStorage.setItem('cah_token', token);
    console.log('🔍 Token set, now checking auth status...');
    checkAuthStatus();
  };

  // Function to clear token and reload
  const clearTokenAndReload = () => {
    console.log('🔍 Clearing token and reloading...');
    localStorage.removeItem('cah_token');
    localStorage.removeItem('user_info');
    setUser(null);
    window.location.reload();
  };

  // Function to validate current token
  const validateCurrentToken = async () => {
    const token = localStorage.getItem('cah_token');
    if (!token || token.trim() === '') {
      console.log('🔍 No token to validate');
      return false;
    }

    try {
      const response = await fetch('/api/auth/me', {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      if (response.ok) {
        const userData = await response.json();
        console.log('🔍 Token is valid, user data:', userData);
        setUser(userData.user);
        return true;
      } else {
        console.log('🔍 Token is invalid, status:', response.status);
        localStorage.removeItem('cah_token');
        setUser(null);
        return false;
      }
    } catch (error) {
      console.error('🔍 Token validation failed:', error);
      return false;
    }
  };

  const updateUserPreferences = async (preferences: Partial<User['preferences']>) => {
    if (!user) return;

    try {
      const token = localStorage.getItem('cah_token');
      const response = await fetch('/api/user/preferences', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify(preferences),
      });

      if (response.ok) {
        const updatedUser = { 
          ...user, 
          preferences: { 
            ...user.preferences, 
            ...preferences,
            // Ensure required properties have default values if undefined
            favorite_personas: preferences.favorite_personas || user.preferences?.favorite_personas || [],
            humor_style: preferences.humor_style || user.preferences?.humor_style || '',
            audience_preference: preferences.audience_preference || user.preferences?.audience_preference || ''
          } 
        };
        setUser(updatedUser);
      }
    } catch (error) {
      console.error('Failed to update preferences:', error);
    }
  };

  const value: UserContextType = {
    user,
    loading,
    login,
    register,
    logout,
    updateUserPreferences,
    refreshAuth,
    debugTokenStatus,
    setTokenManually,
    clearTokenAndReload,
    validateCurrentToken,
  };

  return (
    <UserContext.Provider value={value}>
      {children}
    </UserContext.Provider>
  );
}

export function useUser() {
  const context = useContext(UserContext);
  if (context === undefined) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
}
