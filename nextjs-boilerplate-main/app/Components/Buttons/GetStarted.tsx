
"use client";
import React, { useState } from 'react';
import Lottie from 'react-lottie';
import animationData from '../../assets/arrow.json';


 

const GetStarted = ({ name }) => {
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
      className="inline-flex h-12 items-center justify-center rounded-md border border-gray-400 bg-gray-100 px-6 font-medium text-slate-900 transition-colors focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2 focus:ring-offset-slate-50 hover:bg-gray-300 sm:text-base text-sm"
      onMouseEnter={() => { setIsStopped(false); setIsPaused(false); }}
      onMouseLeave={() => { setIsStopped(true); setIsPaused(true); }}
    
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

export {GetStarted };
