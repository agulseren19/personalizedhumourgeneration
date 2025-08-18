'use client';

import { useEffect, useState } from 'react';
import { useSearchParams } from 'next/navigation';

export default function GoogleCallback() {
  const searchParams = useSearchParams();
  const [status, setStatus] = useState('Processing...');
  const [error, setError] = useState('');

  useEffect(() => {
    const processCallback = async () => {
      try {
        const code = searchParams.get('code');
        const error = searchParams.get('error');

        if (error) {
          setError(`Authentication failed: ${error}`);
          sendMessageToParent('GOOGLE_OAUTH_ERROR', { error });
          return;
        }

        if (!code) {
          setError('No authorization code received');
          sendMessageToParent('GOOGLE_OAUTH_ERROR', { error: 'No authorization code' });
          return;
        }

        // Exchange code for token via backend
        const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
        const response = await fetch(`${backendUrl}/auth/google/callback?code=${code}`, {
          method: 'GET',
          credentials: 'include',
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Authentication failed');
        }

        const data = await response.json();

        if (data.success) {
          setStatus('Authentication successful!');
          // Send success message to parent window
          sendMessageToParent('GOOGLE_OAUTH_SUCCESS', {
            user: data.user,
            access_token: data.access_token,
          });
        } else {
          throw new Error(data.error || 'Authentication failed');
        }

      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error';
        setError(errorMessage);
        sendMessageToParent('GOOGLE_OAUTH_ERROR', { error: errorMessage });
      }
    };

    processCallback();
  }, [searchParams]);

  const sendMessageToParent = (type: string, data: any) => {
    if (window.opener) {
      window.opener.postMessage({ type, ...data }, '*');
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center p-4">
      <div className="bg-gray-800 rounded-lg p-8 max-w-md w-full text-center">
        <div className="mb-6">
          <div className="w-16 h-16 bg-teal-500 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </div>
          <h1 className="text-2xl font-bold text-white mb-2">Google Sign-In</h1>
          <p className="text-gray-300">{status}</p>
        </div>

        {error && (
          <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-4 mb-4">
            <p className="text-red-300 text-sm">{error}</p>
          </div>
        )}

        <div className="text-gray-400 text-sm">
          <p>This window will close automatically.</p>
          <p>If it doesn't close, you can close it manually.</p>
        </div>
      </div>
    </div>
  );
}
