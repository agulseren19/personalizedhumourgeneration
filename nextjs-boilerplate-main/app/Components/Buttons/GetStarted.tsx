
"use client";
import React, { useState } from 'react';
import Lottie from 'lottie-react';
import animationData from '../../assets/arrow.json';


 

const GetStarted = ({ name }: { name: string }) => {
  const [isStopped, setIsStopped] = useState(true);


  return (
    <button
      className="inline-flex h-11 items-center justify-center rounded-md border border-gray-300 bg-white px-5 font-medium text-slate-700 transition-colors focus:outline-none focus:ring-2 focus:ring-slate-300 focus:ring-offset-2 focus:ring-offset-slate-50 hover:bg-gray-50 hover:border-gray-400 sm:text-base text-sm shadow-sm"
      onMouseEnter={() => setIsStopped(false)}
      onMouseLeave={() => setIsStopped(true)}
    
    >
      {name}
      <div className="ml-2 sm:ml-3" style={{ width: '28px', height: '28px' }}>
        <Lottie 
          animationData={animationData}
          loop={false}
          autoplay={false}
          style={{ width: 28, height: 28 }}
          onComplete={() => setIsStopped(true)}
        />
      </div>
    </button>
  );
};

export {GetStarted };
