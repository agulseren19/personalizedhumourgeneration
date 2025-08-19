"use client";
import React, { useState } from 'react';
import Lottie from 'lottie-react';
import animationData from '../../assets/banknote.json';

const Pricing = ({ name, onClick }: { name: string; onClick: () => void }) => {
  const [isStopped, setIsStopped] = useState(true);

  return (
    <button
      className="inline-flex h-12 items-center justify-center rounded-md border border-slate-700 bg-teal-600 px-6 font-medium text-white transition-colors focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2 focus:ring-offset-slate-50 hover:bg-teal-500 sm:text-base text-sm"
      onMouseEnter={() => setIsStopped(false)}
      onMouseLeave={() => setIsStopped(true)}
      onClick={onClick}
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

export default Pricing;
