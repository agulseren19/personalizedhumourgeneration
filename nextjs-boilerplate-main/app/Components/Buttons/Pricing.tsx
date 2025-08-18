"use client";
import React, { useState } from 'react';
import Lottie from 'react-lottie';
import animationData from '../../assets/banknote.json';

const Pricing = ({ name, onClick }) => {
  const [isStopped, setIsStopped] = useState(true);
  const [isPaused, setIsPaused] = useState(true);

  const defaultOptions = {
    loop: false, // Disable looping
    autoplay: false, // Autoplay is set to false
    animationData: animationData,
    rendererSettings: {
      preserveAspectRatio: 'xMidYMid slice'
    }
  };

  return (
    <button
      className="inline-flex h-12 items-center justify-center rounded-md border border-slate-700 bg-teal-600 px-6 font-medium text-white transition-colors focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2 focus:ring-offset-slate-50 hover:bg-teal-500 sm:text-base text-sm"
      onMouseEnter={() => { setIsStopped(false); setIsPaused(false); }}
      onMouseLeave={() => { setIsStopped(true); setIsPaused(true); }}
      onClick={onClick}
    >
      {name}
      <div className="ml-2 sm:ml-3" style={{ width: '30px', height: '30px' }}>
        <Lottie 
          options={defaultOptions}
          height={30} // Reduced height for better fit on mobile
          width={30} // Reduced width for better fit on mobile
          isStopped={isStopped}
          isPaused={isPaused}
        />
      </div>
    </button>
  );
};

export default Pricing;
