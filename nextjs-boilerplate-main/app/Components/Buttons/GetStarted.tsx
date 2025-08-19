
"use client";
import React, { useState } from 'react';
import Lottie from 'lottie-react';
import animationData from '../../assets/arrow.json';


 

const GetStarted = ({ name }: { name: string }) => {
  const [isStopped, setIsStopped] = useState(true);


  return (
    <button
      className="inline-flex h-12 items-center justify-center rounded-md border border-gray-400 bg-gray-100 px-6 font-medium text-slate-900 transition-colors focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2 focus:ring-offset-slate-50 hover:bg-gray-300 sm:text-base text-sm"
      onMouseEnter={() => setIsStopped(false)}
      onMouseLeave={() => setIsStopped(true)}
    
    >
      {name}
      <div className="ml-2 sm:ml-3" style={{ width: '30px', height: '30px' }}>
        <Lottie 
          animationData={animationData}
          loop={false}
          autoplay={false}
          style={{ width: 30, height: 30 }}
          onComplete={() => setIsStopped(true)}
        />
      </div>
    </button>
  );
};

export {GetStarted };
